// Two-stage LightGBM ammonia estimation - ESP32-S3
//
// Stage 1: is ammonia detectable?   (classifier -> probability)
// Stage 2: how much ammonia?        (regressor, only if detectable)
//
// Inputs are RAW sensor values (no scaling needed - tree models are scale-invariant).
// Feature order MUST match training:
//   0 TEMPERATURE, 1 TURBIDITY, 2 DISOLVED OXYGEN, 3 pH, 4 NITRATE

#include <Arduino.h>

#include "lightgbm_runtime.h"
#include "stage1_model.h"
#include "stage2_model.h"

struct Prediction {
    bool  detectable;
    float probability;
    float ammonia;
};

Prediction predict_ammonia(float temperature, float turbidity,
                           float dissolved_oxygen, float pH, float nitrate) {
    float x[5] = {temperature, turbidity, dissolved_oxygen, pH, nitrate};

    // Stage 1: binary classifier -> sigmoid(raw score)
    float raw1 = LightGBMRuntime::raw_score(
        Stage1Model::nodes, Stage1Model::tree_roots, Stage1Model::NUM_TREES, x);
    float prob = LightGBMRuntime::sigmoid(raw1);

    Prediction p;
    p.probability = prob;
    p.detectable  = prob >= 0.5f;
    p.ammonia     = 0.0f;

    // Stage 2: regressor, only when Stage 1 says detectable
    if (p.detectable) {
        float raw2 = LightGBMRuntime::raw_score(
            Stage2Model::nodes, Stage2Model::tree_roots, Stage2Model::NUM_TREES, x);
        p.ammonia = LightGBMRuntime::non_negative(raw2);
    }
    return p;
}

void run_and_print(const char* label, float t, float tu, float doo, float ph, float no3) {
    unsigned long start = micros();
    Prediction p = predict_ammonia(t, tu, doo, ph, no3);
    unsigned long us = micros() - start;

    Serial.printf("%-22s detectable=%s  p=%.3f  ammonia=%.4f  (%lu us)\n",
                  label, p.detectable ? "YES" : "NO", p.probability, p.ammonia, us);
}

void setup() {
    Serial.begin(115200);
    delay(1500);
    Serial.println("\nAmmonia Edge-AI | two-stage LightGBM | ESP32-S3");
    Serial.println("---------------------------------------------");

    // Sample rows taken from the test set (actual ammonia in comments)
    run_and_print("low-ammonia sample",  27.0f,     100, 0.0f,  6.06858f, 272);   // actual ~0
    run_and_print("mid-ammonia sample",  25.0f,      51, 0.0f,  6.44082f, 1078);  // actual ~10.0
    run_and_print("mid-ammonia sample 2", 26.1875f,  51, 2.962f, 6.34095f, 1067); // actual ~17.4
    run_and_print("high-ammonia sample", 26.1875f,   51, 35.851f, 6.42720f, 969); // actual ~53.9
}

void loop() {
    // TODO: replace with live sensor readings
    //   float t  = readTemperature();     // DS18B20
    //   float tu = readTurbidity();
    //   float d  = readDissolvedOxygen();
    //   float ph = readPH();
    //   float n  = readNitrate();
    //   run_and_print("live", t, tu, d, ph, n);
    delay(1000);
}

# GaitIdentifier

A wearable built from an Arduino UNO Q and an MPU-6050 IMU that classifies what the wearer is doing:

- standing still
- walking
- running
- biking (or lying on your back and air-biking)

Two models are compared (no neural nets):

1. **Single-feature model**: uses only the peak acceleration along one axis.
2. **Full-IMU model**: uses all of the IMU outputs (3-axis accelerometer + 3-axis gyroscope).

## Status

Planning. Hardware layout and model design are still being decided.

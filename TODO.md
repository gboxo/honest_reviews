# Dom 11/05



## Review prompt



I am a professional content creator who reviews smartphones. Over the past month, I have thoroughly tested a phone and rated its key features on a scale from 0 to 10, where 0 means very poor and 10 means excellent.

Below is a JSON object containing the phone's details and my feature ratings. Using this data, please write a concise, two-paragraph review of the phone. The review should accurately reflect the ratings but **do not mention the numeric scores explicitly**. Instead, use qualitative descriptions to convey the strengths and weaknesses of the device.

```json
{
  "name": "Zephyr X1",
  "model": "ZX1-2025",
  "brand": "NovaTech",
  "year": 2025,
  "operating_system": "NovaOS 3.2",
  "battery_life": 8,
  "camera_quality": 5,
  "processor_performance": 9,
  "screen_clarity": 8,
  "build_quality": 8,
  "audio_quality": 7,
  "charging_speed": 9,
  "software_experience": 8,
  "multitasking_performance": 8,
  "front_camera_quality": 6,
  "video_recording_quality": 7,
  "fingerprint_sensor_accuracy": 9
}
```

**Instructions:**

- Write a brief review of about two paragraphs.
- Use qualitative language to describe each feature based on the ratings.
- Highlight both strengths and weaknesses.
- Avoid mentioning any numeric scores or exact ratings.
- The tone should be professional and informative.



## Manual Inspection of Outputs

- Qwen 7b is not good enough (doesn't catch subtle info)
- Qwen 32b is good enough



### Create the dataset


1) Ask LLM to generate specification sheets for some products
    - Phone, laptop, tablet, etc.
2) Change the above template
3) Combine the generated data with the template and define a final set of instructions



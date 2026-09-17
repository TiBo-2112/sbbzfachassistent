from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

response = client.chat.completions.create(
    model="qwen/qwen3.5-9b",
    messages=[
        {
            "role": "system",
            "content": "Du bist ein hilfreicher Assistent."
        },
        {
            "role": "user",
            "content": "Antworte exakt mit: LM Studio funktioniert"
        }
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)

from model_setup import model, tokenizer

# 1. Prepare Phi-3 for tests

def test_model(prompt_text):
    messages = [
        {"role": "user", "content": prompt_text},
    ]
    # tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # generation
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=1048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # decoded answers
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return response

# Quick start
print("\n", "Модель готова к тестам!", "\n", "Model is ready for tests!")


# OUTPUT FROM NOTEBOOK: 
# Model is ready for tests!


# 2. Full testing

# Test prompt
messages = [
    {"role": "user", "content": "I am a 60-year-old with hypertension experiencing bloating, rash, abdominal pain for 2 days at night. What could be the cause and what should I do?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
)

# Decode and print
response = tokenizer.batch_decode(outputs)[0]
print(response)

# OUTPUT FROM NOTEBOOK:
#  <|user|> I am a 60-year-old with hypertension experiencing bloating, rash, abdominal pain for 2 days at night. What could be the cause and what should I do?<|end|><|assistant|> Given your symptoms of bloating, rash, and abdominal pain, particularly at night, along with your history of hypertension, there are several potential causes. It's important to consider both gastrointestinal and allergic reactions, as well as the possibility of stress-related factors.
# 1. **Gastrointestinal Causes**: These could include conditions such as gastroesophageal reflux disease (GERD), peptic ulcers, or gastritis, which might be exacerbated by stress or certain foods.
# 2. **Allergic Reactions**: The rash could be a sign of an allergic reaction, possibly to a medication or food.
# 3. **Stress**: Stress can exacerbate symptoms of hypertension and can also cause or worsen gastrointest0inal symptoms.
# Given your age and existing hypertension, it's important to approach this carefully. I recommend the following steps:
#  **Consult a Doctor**: It's essential to get medical advice. Make an appointment with your primary care physician
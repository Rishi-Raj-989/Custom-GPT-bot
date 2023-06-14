import openai

openai.api_key = "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"

keep_prompting = True

while keep_prompting:
    prompt = input('what is your question  ')
    if(prompt == 'exit'):
        keep_prompting = False
    else:
        response = openai.ChatCompletion.create(model = 'gpt-3.5-turbo',messages=[{"role":"user","content":prompt}])
        print(response.choices[0].message.content)

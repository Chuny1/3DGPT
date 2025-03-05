import openai
import json

def modeling_function_call(text_description,function_description, function,
                           function_document,example,max_tokens=2000,temperature=0.3,history=[]):
      """
      Give a short text, call the given functions to generation objects/scene to fit the given text description
      Parameters
      ----------
      text_description: short user given text.
      function_description: short function description.
      function: python code.
      function_document: the detail description of the function.
      example: example of how to use the function.
      max_tokens: max tokens for the detailed text.
      temperature
      Returns
      -------
      response from the agent that contains the function calls.
      """
      if(len(history)==0):
        history = [
          {"role":"system", "content":"You are a good 3D designer who can convert long text descriptions into parameters, and is good at understanding Python functions to manipulate 3D content. "},
        ]
      messages = history

      text = f"""Given the text description: “{text_description}”, we have the following function codes {function}
        Fand the document for function {function_document}.
        Below is an example bout how to make function calls to model the scene to model the scene to fit the description: {example}.
        Understand the function, and model the 3D scene that fits the text description by making a function call."""

      messages.append({"role": "user","content":text})

      modeling_model = openai.ChatCompletion.create(
        model = "gpt-4o-2024-11-20",
        temperature = temperature,
        max_tokens = max_tokens,
        messages = messages
      )

      modeling_text = modeling_model.choices[0].message["content"]
      return modeling_text,messages


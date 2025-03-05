import openai
import json

def conceptulization_call(text_description,info,max_tokens=500,temperature=0,history=[]):
      """
      Give a short text, call the given functions to generation objects/scene to fit the given text description
      Parameters
      ----------
      text_description: short user given text.
      info: information required to be queried or imagine. The information required for parameter inference.
      max_tokens: max tokens for the detailed text.
      temperature
      Returns
      -------
      response from the agent that contains the function calls.
      """
      if(len(history)==0):
        history = [
          {"role":"system", "content":"You are a skilled writer, especially when it comes to describing the appearance of objects and large scenes"},
        ]
      messages = history

      text = f"""Given a text description "{text_description}", provide detailed descriptions for the following information: "{info}". 
        For term not mentioned in the description, use your imagination to ensure they fit the text description."""

      messages.append({"role": "user","content":text})



      conceptualization_augmentation_model = openai.ChatCompletion.create(
        model = "gpt-4o-2024-11-20",
        temperature = temperature,
        max_tokens = max_tokens,
        messages = messages
      )

      conceptualization_text = conceptualization_augmentation_model.choices[0].message["content"]
      return conceptualization_text,messages
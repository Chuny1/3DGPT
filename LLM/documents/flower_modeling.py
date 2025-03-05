import sys
import os
import random
import sys
import os
import re
import openai
import json
import ast
# setting path
sys.path.append('../../LLM')
from agents.modeling_agent import modeling_function_call
from agents.conceptulization_agent import conceptulization_call
import openai
import json
import numpy as np

center_rad = 0.1
length = center_rad * 3
base_width = length * 0.7
wrinkle = base_width * 0.001
min_petal_angle = np.deg2rad([80])[0]
max_petal_angle = np.deg2rad([90])[0]
curl = np.deg2rad([40])[0]
petal_karg = {
    'center_rad': center_rad,
    'seed_size': 0.006,  # (0.005, 0.01)
    'min_petal_angle': min_petal_angle,  # np.deg2rad(np.sort(uniform(-20, 100, 2)))
    'max_petal_angle': max_petal_angle,
    'wrinkle': wrinkle,  # (0.003, 0.02)
    'curl': curl,  # np.deg2rad(normal(30, 50))
    'base_width': 0.06,
    'top_width': 0.006707742,
    'length': 0.14,
}
red_rose = {
    'petal_number': 40,
    'petal_length': length,
    'petal_width': base_width,
    'petal_roundness': 0,
    'petal_color': [235 / 255, 9 / 255, 9 / 255, 1],
    'open': 0.4,
    'petal': petal_karg,
}


def get_flower_modeling_documents():

    usage = """
    model the 3D flower. 
    """

    info = """
    flower center size, flower petal appearance, including the petal length, petal width, petal color, the ratio of the petal length and width, does the petal has large wrinkle? maximum petal numbers.
    the maximum petal angle, flower openess. 
    """

    code = """
    import FlowerFactory 
    def model_flower(center_size, petal_length,petal_width,petal_roundness,min_petal_angle,max_petal_angle,petal_wrinkle,petal_color,petal_numbers): 
        flower_center = FlowerFactory.generate_flower_center(center_size) # generate flower center
        petals = FlowerFactory.generate_petal_shape(petal_length, petal_width, petal_roundness, min_petal_angle,max_petal_angle,petal_wrinkle,petal_curl) # generate petal
        petals = FlowerFactory.color_petal(petal,petal_color) # color the petal 
        flower = FlowerFactory.generate_flower(petals,petal_numbers,flower_center) # combine the flower center and petals 
        
        return  flower
    """

    document = """
    Explanation: This function takes flower control parameters as input and models a 3D flower as output.
    Inputs:  
    - center_size (float): The size of the flower center.
    - petal_length (float): The length of a petal.
    - petal_width (float): The width of a petal.
    - petal_roundness (float): The roundness of a petal, ranging from 0 to 1. A value of 0 represents a very round petal shape, while 1 represents a sharp petal shape. For example, a rose has a roundness of around 0.3, and a water lily has a roundness of around 0.7 (sharp at the top).
    - min_petal_angle (float): The minimum angle for petals in degrees. 0 degrees is horizontal to the flower center, 
    90 degrees is perpendicular to the flower center, and angles larger than 90 degrees point inward toward the flower center. 
    Angles less than 0 degrees point toward the ground. Small angle for unblossomed flower, large angle for fully openned flower. 
    - max_petal_angle (float): The maximum petal angle in degrees. Similar to min_petal_angle.
    - petal_wrinkle (float): The size of wrinkles on the petal. For example, if the petal width is 1 and petal_wrinkle is 0.1, it will create 10 wrinkles per petal.
    - petal_color (np.array(3)): The RGB color of the flower petal.
    - petal_numbers (int): The number of petals in the flower.
    Outputs: 3D flower
    """

    example = """
    Question: given the text description of the flower: “The "white rose" is a beautiful and classic flower known for its elegance and symbolism. Let's extend the description of a white rose in detail, considering various aspects:
    
    1. Center Size:
    The center of a white rose, often referred to as the "heart" of the flower, is typically small, measuring around 1-2 centimeters in diameter. It forms the focal point from which the petals radiate outward.
    2. Petal Length:
    White rose petals can vary in length but generally range from 2 to 3.5 inches (5 to 9 centimeters). 
    4. Petal Width:
    The width of each white rose petal is approximately 2-3 centimeters, making them relatively narrow compared to their length.
    6. Petal Roundness:
    White rose petals are gently curved and exhibit a soft, rounded shape. They lack the pointed serrations seen in some other rose varieties, giving them a more gentle and serene appearance.
    7. Openness of the Flower:
    White roses typically have a moderately open bloom, with petals that are slightly unfurled but still embrace the center. This openness allows for a glimpse of the intricate structure within the heart of the flower.
    8. Petal Wrinkle Appearance:
    The petals of a white rose appear smooth and unblemished, with a satiny texture that adds to their pristine beauty. 
    9. Petal Color:
    white roses are characterized by their pure, snowy-white petals. 
    10. Typical Petal Numbers:
    White roses typically have five petals, although it's not uncommon to find varieties with more or fewer petals. The five-petal arrangement is symmetrical and contributes to the rose's balanced and harmonious appearance.” analysis the function parameter and call the function to model the flower that fits the text description.  
        Solution: From above description of the flower, we can know that the center size of white rose is around 1-2 cm, let's set 'center_size' to 1.4,
        the petal length is around 5-9 cm and compare to the center size, it noticeably longer, as we set 'center_size' to 1.4, let's set 'petal length' as 6,
        the petal width is around 2-3 cm let's set the 'petal_width' as 3. "genetly curved and exhibit a soft, rounded shape" indicate the 'petal_roundness' value should be low, we can set 'petal_roundness' to 0.3.
        the "White roses typically have a moderately open bloom, with petals that are slightly unfurled but still embrace the center" indicate the 'min_petal_angle' for some flower can be slightly open, we can set 'min_petal_angle' to 60. 
        To allow some petals embrace the center, let's set the 'max_petal_angle' to 90. "The petals of a white rose appear smooth and unblemished" indicate the 'petal_wrinkle' should have a very small value, we can set 'petal_wrinkle' to 0.001.
        While flower has the snowy-white petals, the 'petal_color' can set as (250,249,239), which is a nearly white color. 
        as describe in the text, the petal number is typically be five, let's set it to 5. 
        ```python model_flower(center_size=1.4, petal_length=6,petal_width=3,
        petal_roundness=0.3,min_petal_angle=60,max_petal_angle=90,petal_wrinkle=0.001,
        petal_color=[250,249,239],petal_numbers=5)
        ```  
    """

    return usage,info, code,document,example


def flower_modelling_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    # Define the regex pattern to match the content between '' and 'ddere'
    pattern = r'model_flower\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'

    counter = 0
    # set default dictionary to handle wrong response
    dictionary = {"flower_param": None}

    try:
        # Loop through the matches and print each one
        for match in matches:
            counter += 1
            extracted_text = match.group(1)
            pattern = r',(?![^\[]*\])'  # Define the regex pattern to match commas outside of square brackets
            result = re.split(pattern, extracted_text)  # Use re.split to split the string based on the pattern
            parameters = [s.strip() for s in result]  # Trim spaces from the resulting strings
            parameter_length = len(parameters)

            print('parameters',parameters)

            if (parameter_length != 9):
                state = 'Parameter length not correct'
                break

            center_size = float(parameters[0].split('=')[1])
            petal_length = float(parameters[1].split('=')[1])
            petal_width = float(parameters[2].split('=')[1])
            petal_roundness = float(parameters[3].split('=')[1])
            min_petal_angle = float(parameters[4].split('=')[1])
            max_petal_angle = float(parameters[5].split('=')[1])
            petal_wrinkle =  float(parameters[6].split('=')[1])
            petal_color_rgb =  str(parameters[7].split('=')[1]).replace('[','').replace(']','').split(',')
            petal_color_r,petal_color_g, petal_color_b = max(0,min(255,int(petal_color_rgb[0]))), \
                max(0,min(255,int(petal_color_rgb[1]))), max(0,min(255,int(petal_color_rgb[2])))
            petal_color = (petal_color_r,petal_color_g,petal_color_b)
            petal_numbers = int(parameters[8].split('=')[1])

            check_data_type = isinstance(center_size, float) and isinstance(petal_length, float) and isinstance(petal_width, float) \
                              and isinstance(petal_roundness, float) and isinstance(min_petal_angle, float) and isinstance(max_petal_angle,float) \
                              and isinstance(petal_wrinkle, float) and isinstance(petal_color_r, int) and isinstance(petal_color_g,int) \
                              and isinstance(petal_color_b, int) and isinstance(petal_numbers,int)

            if (not check_data_type):
                state = 'Data type error'
                return state, None

            flower_params = {
                "center_size":center_size,
                "petal_length":petal_length,
                "petal_width" :petal_width,
                "petal_roundness" : petal_roundness,
                "min_petal_angle" : min_petal_angle,
                "max_petal_angle" : max_petal_angle,
                "petal_wrinkle" : petal_wrinkle,
                "petal_color" : petal_color,
                "petal_numbers":petal_numbers
                }

            dictionary['flower_params'] = flower_params
            break

    except:
        state = 'No match found'
        return state, None

    if (counter == 0):
        state = 'No match found'
        return state, None

    return state, dictionary



if __name__ == "__main__":
    usage,info, code,document,example = get_flower_modeling_documents()
    text_description = "gerbera"
    augmented_text, _ =  conceptulization_call(text_description,info)
    print('augmented text',augmented_text)
    text_description = text_description + augmented_text # append augmented_text to the initial text description.
    text,_ = modeling_function_call(text_description=text_description,function_description=usage,function_document=document,function=code,example=example)
    print('modeling result:',text)
    state, dictionary = flower_modelling_parser(text)
    print('state:',state,'flower dictionary:', dictionary)

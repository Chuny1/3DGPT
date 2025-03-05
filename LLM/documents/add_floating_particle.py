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

def get_add_floating_particle_documents():
    usage = """
    add floating particles to the 3D scene by control blender using python code 
    """

    info = """
        is the scene suitable to have falling leaves? is the scene in raining? is suitable to have dust on the air? is the scene have snow falling?
    """

    code = """
    def add_particles(scene, has_leaf_particles, has_rain_particles, has_dust_particles, has_marine_snow_particles,
                      has_snow_particles):
        if (has_leaf_particles):
            scene.apply_leaf_particles()
        if (has_rain_particles):
            scene.apply_rain_particles()
        if (has_dust_particles):
            scene.apply_dust_particles()
        if (has_marine_snow_particles):
            scene.apply_marine_snow_particles()
        if (has_snow_particles):
            scene.apply_snow_particles()
        return
    """




    document = """
    The function takes built natural scene as input, it will add floating particles on the air.  
    Inputs:  
    scene: the built natural scene.  
    has_leaf_particles: (bool) whether should add leaf particles on the air. It more likely set to be True when windy day and season is autumn.   
    has_rain_particles: (bool) whether should add rain particles on the air. It can set to be True when the scene is likely rainy.  
    has_dust_particles: (bool) whether should add dust particles on the air. It more likely set to be True when the air condition is not good and when the scene contains dust.  
    has_marine_snow_particles: (bool) whether should add marine snow particles on the air. It can set to be True when the scene is likely have large snow.  
    has_snow_particles: (bool) whether should add snow particles on the air. It can set to be True when the scene is likely have small snow. 
    """

    example = """
    Question: given the text description of the scene: “A dense pine forest blanketed in a fresh layer of snow, as delicate snowflakes continue to fall” analysis the function parameter and call the function to generate the sky.  
    Solution: From the scene description, it is a winter snowy day, it is reasonable to the falling leaves, we can choose set 'has_leaf_particles' to be True to increase the scene diversity. As it is a snowy day, it is very uncommon to have rain together, we choose set 'has_rain_particles' to be False. Snowy day should has clean air, so we choose set 'has_dust_particles' to be False. From the description of “delicate snowflakes continue to fall”, the scene is likely has marine snow, we can set ‘has_marine_snow_particles’ to be True. As we already choose to have marine snow on the scene, we choose set 'has_snow_particle' to be False. Based on above analysis, we can adding particles to the scene by calling following functions:  
    ```python add_particles(scene,has_leaf_particles=False, has_rain_particles=False, has_dust_particles=False, has_marine_snow_particles=True, has_snow_particles = False)```  
    """

    return usage,info,code,document,example



def add_floating_particle_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    pattern = r'add_particles\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'
    counter = 0
    # set default dictionary to handle wrong response

    dictionary = {
        "has_leaf_particles": False,
        "has_rain_particles": False,
        "has_dust_particles": False,
        "has_marine_snow_particles": False,
        "has_snow_particles": False
    }

    # Loop through the matches and print each one
    for match in matches:
        counter += 1
        extracted_text = match.group(1)
        parameters = extracted_text.split(',')
        print('parameters',parameters)

        parameter_length = len(parameters)
        print(parameter_length)
        if(parameter_length!=6):
            state = 'Parameter length not correct'
            break
        parameters = parameters[1:]
        has_leaf_particles = ('True' in parameters[0]) or ('true' in parameters[0])
        has_rain_particles = ('True' in parameters[1]) or ('true' in parameters[1])
        has_dust_particles = ('True' in parameters[2]) or ('true' in parameters[2])
        has_marine_snow_particles = ('True' in parameters[3]) or ('true' in parameters[3])
        has_snow_particles = ('True' in parameters[4]) or ('true' in parameters[4])

        dictionary["has_leaf_particles"] = has_leaf_particles
        dictionary["has_rain_particles"] = has_rain_particles
        dictionary["has_dust_particles"] = has_dust_particles
        dictionary["has_marine_snow_particles"] = has_marine_snow_particles
        dictionary["has_snow_particles"] = has_snow_particles

        break

    if(counter==0):
        state = 'No match found'
        return state, None

    return state, dictionary



if __name__ == "__main__":
    usage,info,code,document,example = get_add_floating_particle_documents()

    text_description = "The beach, a tapestry of sun-kissed sand, stretched out for miles, caressed by the gentle waves that rhythmically lapped against the shore, as seagulls soared and dove in graceful arcs above the sparkling sea."

    augmented_text, _ =  conceptulization_call(text_description,info)
    print('augmented text',augmented_text)
    text_description = text_description + augmented_text # append augmented_text to the initial text description.


    text,_ = modeling_function_call(text_description=text_description,function_description=usage,function_document=document,function=code,example=example)
    state, dictionary = add_floating_particle_parser(text)
    print(text)
    print('state',state)
    print('dictionary',dictionary)

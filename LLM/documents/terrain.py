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

def get_terrain_modelling_documents():
    usage = """
    create the base terrain mesh.
    """

    info = """
    possible terrain for the scene, e.g. arctic,canyon,cave,cliff,coast,coral_reef,desert,forest,mountain,plain,river. 
    """

    code = """
    ```python
    import Terrain # import pre-built terrain_element 
    def generate_terrain(arctic,canyon,cave,cliff,coast,coral_reef,desert,forest,mountain,plain,river):
    
    # Create a dictionary to map parameter names to their values
    terrain_params = {
        'arctic': arctic,
        'canyon': canyon,
        'cave': cave,
        'cliff': cliff,
        'coast': coast,
        'coral_reef': coral_reef,
        'desert': desert,
        'forest': forest,
        'mountain': mountain,
        'plain': plain,
        'river': river
    }

    # Find the terrain with the largest value
    terrain_name = max(terrain_params, key=lambda x: terrain_params[x])
    mesh = TerrainElements.get(terrain_name) # build the terrain mesh with selected terrain type
    return mesh 
    ```
    """

    document = """
    Explanation: This function takes possibility of each terrain elements as input, build the terrain with
     the element with the largest probability.  
    Inputs:
    arctic(float): probability that the scene to be arctic.
    canyon(float): probability that the scene to be canyon.
    cave(float): probability that the scene to be cave.
    cliff(float): probability that the scene to be cliff.
    coast(float): probability that the scene to be coast.
    coral_reef(float): probability that the scene to be coral_reef.
    desert(float): probability that the scene to be desert.
    forest(float): probability that the scene to be forest.
    mountain(float): probability that the scene to be mountain.
    plain(float): probability that the scene to be plain.
    river(float): probability that the scene to be river.
    """

    example = f"""
    Question: given the text description of the scene: "A cascading waterfall surrounded by dense, ancient forest."
    Analysis the function parameter and call the function to {usage}.
    Solution: 
    From the text description, if the "waterfall" exist, there are less chance the scene become 'artic','canyon','coast','coral_reef','desert',and 'plain',
    we can choose set 0 to those parameter. The cave and cliff is more likely to exist to have waterfall. In the description, it is a 
    cascading waterfall, so we can assign higher value to 'cliff' than 'cave'. The river and forest also make sense for the scene description as waterfall may near the river and the description indicate there are forest nearby.
    We can assign high value to 'river' and 'forest' as well. As forest have been directly mentioned in the text, we should assign highest value to 'forest'. 
    We choose set parameters as following values:
    ```python generate_terrain(arctic=0,canyon=0,cave=0.5,cliff=0.8,coast=0,coral_reef=0,desert=0,forest=0.9,mountain=0,plain=0,river=0.7)```
    """
    return usage, info, code, document, example


def terrain_modelling_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    # Define the regex pattern to match the content between '' and 'ddere'
    pattern = r'generate_terrain\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'

    counter = 0
    # set default dictionary to handle wrong response
    dictionary = {"terrain": None}

    try:
        # Loop through the matches and print each one
        for match in matches:
            counter += 1
            extracted_text = match.group(1)
            pattern = r',(?![^\[]*\])'  # Define the regex pattern to match commas outside of square brackets
            result = re.split(pattern, extracted_text)  # Use re.split to split the string based on the pattern
            parameters = [s.strip() for s in result]  # Trim spaces from the resulting strings
            parameter_length = len(parameters)
            if (parameter_length != 11):
                state = 'Parameter length not correct'
                break

            arctic = float(parameters[0].split('=')[1])
            canyon = float(parameters[1].split('=')[1])
            cave = float(parameters[2].split('=')[1])
            cliff = float(parameters[3].split('=')[1])
            coast = float(parameters[4].split('=')[1])
            coral_reef = float(parameters[5].split('=')[1])
            desert =  float(parameters[6].split('=')[1])
            forest =  float(parameters[7].split('=')[1])
            mountain = float(parameters[8].split('=')[1])
            plain = float(parameters[9].split('=')[1])
            river = float(parameters[10].split('=')[1])

            check_data_type = isinstance(arctic, float) and isinstance(canyon, float) and isinstance(cave, float) \
                              and isinstance(cliff, float) and isinstance(coast, float) and isinstance(coral_reef,float) \
                              and isinstance(desert, float) and isinstance(forest, float) and isinstance(mountain,float) \
                              and isinstance(plain, float) and isinstance(river,float)

            if (not check_data_type):
                state = 'Data type error'
                return state, None

            terrain_params = {
                'arctic': arctic,
                'canyon': canyon,
                'cave': cave,
                'cliff': cliff,
                'coast': coast,
                'coral_reef': coral_reef,
                'desert': desert,
                'forest': forest,
                'mountain': mountain,
                'plain': plain,
                'river': river
            }

            # Find the terrain with the largest value
            terrain_name = max(terrain_params, key=lambda x: terrain_params[x])
            terrain = {}
            terrain['base'] = terrain_name
            dictionary['terrain'] = terrain
            break

    except:
        state = 'No match found'
        return state, None

    if (counter == 0):
        state = 'No match found'
        return state, None

    return state, dictionary


def get_terrain_surface_documents():
    usage = """
    set the terrain surface material 
    """

    info = """possible surface appearance for the mountain, ground, choose from
     ['snow','ice','mud','sand','mountain','cobble_stone','cracked_ground','dirt','stone','soil','chunkyrock']. Possible 
     state for liquid, choose from ['water','ice']"""

    code = """
    ```python
    def add_terrain_material(mountain_elements, ground_elements, liquid_elements, mountain_material, ground_material, liquid_material):
        for mountain_element in mountain_elements:
            mountain_element.apply(mountain_material) # apply mountain material for all mountain elements
        for ground_element in ground_elements:
            ground_element.apply(ground_material) # apply ground material for all ground elements
        for liquid_element in liquid_elements:
            liquid_element.apply(liquid_material) # apply liquid material for all liquid elements
        return 
    ```
    """

    document = """
    Explanation: This function takes the terrain elements and elements surface material as input, apply material to elements. 
    Inputs:
    mountain_elements: terrain elements list of mountain. 
    ground_elements: terrain elements list of ground. 
    liquid_elements: terrain elements list of liquid.  
    mountain_material(str): the surface material for mountain elements. Must choose from ['snow','ice','mud','sand','mountain','cobble_stone','cracked_ground','dirt','stone','soil','chunkyrock'].
    ground_material(str): the surface material for ground elements. Must choose from ['snow','ice','mud','sand','mountain','cobble_stone','cracked_ground','dirt','stone','soil','chunkyrock'].
    liquid_material(str): the liquid material for liquid elements. Must choose from ['water','ice']
    """

    example = f"""
    Question: given the text description of the scene: "The river, reflecting the clear blue of the sky, glistened like a silver ribbon as it wound its way through the lush valley, its tranquil waters whispering secrets to the ancient trees."
    Analysis the function parameter and call the function to {usage}.
    Solution: 
    From the text description, we can know that the mountain and ground has trees, in the material list ['snow','ice','mud','sand','mountain','cobble_stone','cracked_ground','dirt','stone','soil','chunkyrock'], ['mud','mountain'] could
    be suitable choice for the mountain and ground material. We choose set 'mountain_material' as 'mountain' and 'ground_material' as 'mud'. The scene seem not in winter, so the river may not frozen, we can choose to set the 'liquid_material' as 'water'. 
    We can call the function to apply the material:
    ```python add_terrain_material(mountain_elements, ground_elements, liquid_elements, mountain_material = 'mountain', ground_material ='mud', liquid_material = water)```
    """

    return usage, info, code, document, example


def terrain_surface_parser(text):
    """
        get the agent response return the response state and extracted information
    """
    # Define the regex pattern to match the content between '' and 'ddere'
    pattern = r'add_terrain_material\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'

    counter = 0
    # set default dictionary to handle wrong response
    dictionary = {
        "ground_collection": "dirt",
        "mountain_collection": "mountain",
        "liquid_collection": "water"
    }
    try:
        # Loop through the matches and print each one
        for match in matches:
            counter += 1
            extracted_text = match.group(1)
            pattern = r',(?![^\[]*\])'  # Define the regex pattern to match commas outside of square brackets
            result = re.split(pattern, extracted_text)  # Use re.split to split the string based on the pattern
            parameters = [s.strip() for s in result]  # Trim spaces from the resulting strings
            parameter_length = len(parameters)
            if (parameter_length <= 3):
                state = 'Parameter length not correct'
                break

            mountain_material, ground_material, liquid_material = '', '', ''
            print(parameters)
            if (len(parameters) >= 4):
                mountain_material = parameters[3].split('=')[1].replace('\'', '')
            if (len(parameters) >= 5):
                ground_material = parameters[4].split('=')[1].replace('\'', '')
            if (len(parameters) >= 6):
                liquid_material = parameters[5].split('=')[1].replace('\'', '')

            def is_sublist(a, b):
                return set(b).issubset(set(a))

            surface_list = ['ice', 'snow', 'mud', 'sand', 'mountain', 'cobble_stone',
                            'cracked_ground', 'dirt', 'stone', 'soil', 'chunkyrock', '']
            water_surface_list = ['ice', 'water', '']

            check_data_type = is_sublist(surface_list, [mountain_material]) and \
                              is_sublist(surface_list, [ground_material]) and \
                              is_sublist(water_surface_list, [liquid_material])

            if (not check_data_type):
                state = 'Data type error'
                return state, None

            if (len(ground_material) > 0):
                dictionary['ground_collection'] = ground_material
            else:
                dictionary['ground_collection'] = random.choice(surface_list)

            if (len(mountain_material) > 0):
                dictionary['mountain_collection'] = mountain_material
            else:
                dictionary['mountain_collection'] = random.choice(surface_list)

            if (len(liquid_material) > 0):
                dictionary['liquid_collection'] = liquid_material
            else:
                dictionary['liquid_collection'] = random.choice(water_surface_list)

            break

    except:
        state = 'No match found'
        return state, None

    if (counter == 0):
        state = 'No match found'
        return state, None

    return state, dictionary


if __name__ == "__main__":
    text_description_list = [
        "Dawn's mist cloaks a pine forest. Needles glisten with dew, and the soft ground is carpeted with pine needles.",
        "Crisp autumn day; trees display vibrant foliage against a clear blue sky. Leaves rustle on pathways.",
        "Sunset paints the sky on a tropical beach. Palm silhouettes sway, while warm sand meets lapping waves.",
        "Heavy rain drenches a mountain pass. Dark trees stand against a gray backdrop; the rocky terrain glistens.",
        "A forest clearing blanketed in snow. White-barked birch trees stand tall under a gray sky."]

    usage, info, code, document, example = get_terrain_modelling_documents()
    text_description = text_description_list[4]

    augmented_text, _ = conceptulization_call(text_description, info)
    print('augmented text:', augmented_text)
    print('*' * 8)
    text_description = text_description + augmented_text  # append augmented_text to the initial text description.
    text, _ = modeling_function_call(text_description=text_description, function_description=usage,
                                     function_document=document, function=code, example=example)

    print("modeling response:", text)
    print('*' * 8)
    state, dictionary = terrain_modelling_parser(text)
    print('state',state,'modeling dictionary:', dictionary)
    print('*' * 8)

    usage, info, code, document, example = get_terrain_surface_documents()
    text, _ = modeling_function_call(text_description=text_description, function_description=usage,
                                     function_document=document, function=code, example=example)

    print("surface response:", text)
    print('*' * 8)
    state, dictionary = terrain_surface_parser(text)
    print('surface dictionary:', dictionary)


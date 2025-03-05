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

def get_add_elements_documents():

    usage = """
    add elements to the 3D scene
    """

    info = """
        is that reasonable for trees in the scene, is that reasonable for rocks in the scene, should have grass in the
        scene, is that suitable to add monocots, ferns, flowers to the scene?
    """

    code = """
    def add_elements(scene, add_clouds, add_trees, add_rocks, add_grass, add_monocots, add_ferns, add_flowers, add_pine_needles,
                     add_decoractive_plants, add_chopped_trees, add_ground_leaves, add_snow):
        
        if(add_clouds):
            scene.add_clouds() # add clouds to the scene
        if(add_trees):
            scene.add_trees() # add trees to the scene 
        if (add_rocks):
            scene.add_rocks() # add rock to the scene 
        if (add_grass):
            scene.add_grass() # add grass to the scene
        if (add_monocots):
            scene.add_monocots() # add monocots to the scene
        if (add_ferns):
            scene.add_ferns() # add ferns to the scene
        if (add_flowers):
            scene.add_flowers() # add flowers to the scene
        if (add_pine_needles): 
            scene.add_pine_needles() # add pine needles to the scene
        if (add_decoractive_plants):
            scene.add_decoractive_plants() # add decoractive plants to the scene
        if (add_chopped_trees):
            scene.add_chopped_trees() # add chopped trees to the scene
        if (add_ground_leaves):
            scene.add_ground_leaves() # add ground leaves to the scene
        if (add_snow):
            scene.add_snow() # add snow to the scene 
        
        return
    """

    document = """
    Explanation:  The function takes built natural scene as input, it will add element to enrich the scene.  
    
    Inputs:  
    scene: the built natural scene.
    add_clouds: (bool) whether should add clouds to the scene.
    add_trees: (bool) whether should add trees to the scene.   
    add_rocks: (bool) whether should add rock on the scene.  
    add_grass: (bool) whether should add grass on the scene.  
    add_monocots: (bool) whether should add monocots on the scene.  
    add_ferns: (bool) whether should add ferns on the scene.   
    add_flowers: (bool) whether should add flowers on the scene.  
    add_pine_needles: (bool) whether should add pine needles on the scene. 
    add_decoractive_plants: (bool) whether should add decorative plants on the scene. 
    add_chopped_trees: (bool) whether should add chopped trees on the scene.  
    add_ground_leaves: (bool) whether should add ground_leaves on the scene. 
    add_snow: (bool) whether should add snow on the scene. 
    """

    example = """
    Question: given the text description of the scene: “A dense pine forest blanketed in a fresh layer of snow, as delicate snowflakes continue to fall” analysis the function parameter and call the function to add more elements that make the scene looks more rich.  
    Solution: As the forest "blanketed in a fresh layer of snow", it more likely be the weather right after snowy, we choose not add clouds to make the sky more clean. As it in the “pine forest”, it mush have trees, the 'add_trees'  the rock can be exist, we can set ‘add_rock’ to be True to make the scene more rich. From the description of “blanketed in a fresh layer of snow”, the scene is likely covered by heavy snow, we may not able to see the grass, we can choose set ‘add_grass’ to False. The monocots, ferns and flowers are less likely appear on winter scene, we choose set them all to be False. As it is “pine forest”, we can add more pine needles to the scene, we choose set ‘add_pine_needles’ to True. Decorative plants, chopped trees and ground levaes are reasonable to be add to the scene to increase the diversity and richness. We choose set them all to True. We can identify 'snow' in the text description, so we must set 'add_snow' to True.  
    ```python add_elements(scene, add_clouds = True, add_trees = True, add_rocks = True,add_grass=False,add_monocots=Flase,add_ferns=Flase,add_flowers=False,add_pine_needles=True,add_decoractive_plants=True,add_chopped_trees=True,add_ground_leaves=True,add_snow=True)```  
    """
    return usage,info, code,document,example


def add_elements_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    pattern = r'add_elements\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'
    counter = 0
    # set default dictionary to handle wrong response
    dictionary = {
        "add_rock": False,
        "add_grass": False,
        "add_monocots": False,
        "add_ferns": False,
        "add_flowers": False,
        "add_pine_needle": False,
        "add_decorative_plants": False,
        "add_chopped_trees": False,
        "add_ground_leaves": False,
        "add_snow":False,
    }

    # Loop through the matches and print each one
    for match in matches:
        counter += 1
        extracted_text = match.group(1)
        parameters = extracted_text.split(',')
        print('parameters',parameters)

        parameter_length = len(parameters)
        print(parameter_length)
        if(parameter_length!=13):
            state = 'Parameter length not correct'
            break
        parameters = parameters[1:]
        add_clouds = ('True' in parameters[0]) or ('true' in parameters[0])
        add_trees = ('True' in parameters[1]) or ('true' in parameters[1])
        add_rocks = ('True' in parameters[2]) or ('true' in parameters[2])
        add_grass = ('True' in parameters[3]) or ('true' in parameters[3])
        add_monocots = ('True' in parameters[4]) or ('true' in parameters[4])
        add_ferns = ('True' in parameters[5]) or ('true' in parameters[5])
        add_flowers = ('True' in parameters[6]) or ('true' in parameters[6])
        add_pine_needles = ('True' in parameters[7]) or ('true' in parameters[7])
        add_decoractive_plants = ('True' in parameters[8]) or ('true' in parameters[8])
        add_chopped_trees = ('True' in parameters[9]) or ('true' in parameters[9])
        add_ground_leaves = ('True' in parameters[10]) or ('true' in parameters[10])
        add_snow = ('True' in parameters[11]) or ('true' in parameters[11])

        dictionary["add_rock"] = add_rocks
        dictionary["add_grass"] = add_grass
        dictionary["add_monocots"] = add_monocots
        dictionary["add_ferns"] = add_ferns
        dictionary["add_flowers"] = add_flowers
        dictionary["add_pine_needle"] = add_pine_needles
        dictionary["add_decorative_plants"] = add_decoractive_plants
        dictionary["add_chopped_trees"] = add_chopped_trees
        dictionary["add_ground_leaves"] = add_ground_leaves
        dictionary["add_snow"] = add_snow

        break

    if(counter==0):
        state = 'No match found'
        return state, None

    return state, dictionary


if __name__ == "__main__":
    usage,info,code,document,example = get_add_elements_documents()

    text_description = "The beach, a tapestry of sun-kissed sand, stretched out for miles, caressed by the gentle waves that rhythmically lapped against the shore, as seagulls soared and dove in graceful arcs above the sparkling sea."
    augmented_text, _ =  conceptulization_call(text_description,info)
    print('augmented text',augmented_text)
    text_description = text_description + augmented_text # append augmented_text to the initial text description.

    text,_ = modeling_function_call(text_description=text_description,function_description=usage,function_document=document,function=code,example=example)
    state, dictionary = add_elements_parser(text)
    print('modeling output',text)
    print('state',state)
    print('dictionary',dictionary)





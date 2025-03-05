import sys
import os
import random
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

def get_tree_modeling_documents():

    usage = """
    add trees to the 3D scene
    """

    info = """
    if the given text is describe the tree, provide tree density, tree leaf type, any fruit in the tree? if has fruit, what kind of fruit in the tree? leaf density.
    else, imagine what type of trees is suitable for the scene. provides the tree descriptions in detail first, then provide tree density, tree leaf type, any fruit in the tree? if has fruit, the fruit in the tree, choose from [’apple’, ’blackberry’, ’coconutgreen’, ’durian’, ’starfruit’, ’strawberry’ ,’None’] ,leaf density for each tree. 
    """

    code = """
    import TreeFactory 
    def add_trees(scene, density, distance_min, leaf_type, fruit_type, leaf_density): 
        trees = TreeFactory.create(density, leaf_type, fruit_type, leaf_density) # create trees 
        scene.placement(trees, distance_min) # place trees to the scene 
        return  
    """

    document = """
    Explanation: The function takes built natural scene as input, it will add tree to the scene.   
    Inputs:  
    scene: the built natural scene.  
    density(float): the density of the trees. distribution from 0.002 to 0.05. 0.002 only few trees, 0.01 garden-like density, 0.04 forest-like density. 
    distance_min(float): minimum distance between trees.  
    leaf_type(string): the type of leaf on the tree, select one from the list [‘leaf’, ’leaf_broadleaf’, ’leaf_ginko’, ’leaf_maple’, ’flower’, ‘None’]. Here ‘leaf’ is the general leaf type that allow further custom setting, e.g. changing leaf shape, color. ‘leaf_broadleaf’, ’leaf_ginko’’ and ‘leaf_maple’ will built the leaf with predefined shape with broad leaf, ginko leaf and maple leaf. ‘flower’ will create flowers on the tree instead of leaves.  ’None’ will not generate leaves on the tree. 
    fruit_type(string): the type of fruit on the tree, select one from the list [’apple’, ’blackberry’, ’coconutgreen’, ’durian’, ’starfruit’, ’strawberry’ ,’None’]. ’apple’, ’blackberry’, ’coconut_green’, ’durian’, ’starfruit’, ’strawberry’ will create corresponding fruit on the tree. ‘’custom_fruit’ can create customisable fruit on the tree with further adjustment, if the desired fruit cannot be found on the above list, we should choose ‘custom_fruit’. ‘None’ will not generate any fruit on the tree.  
    """

    example = """
    Question: given the text description of the scene: “ Graceful and slender, a cluster of birch trees graces the garden with its presence. Their smooth, silvery-white bark contrasts beautifully with the vibrant autumn foliage that adorns their delicate branches. As the sunlight filters through their leaves, the scene is painted with a gentle, ethereal glow. The birch trees create a soothing rustling sound as their leaves quiver in the autumn breeze. Positioned near a pathway, a dogwood tree catches the eye with its striking red stems and a burst of brilliant red and burgundy leaves. Clusters of small, vibrant berries cling to its branches, providing a feast for passing birds. The dogwood's foliage forms a mesmerizing display of colors, transitioning from deep reds at the tips to shades of orange and even a touch of purple nearer to the trunk. Nestled in a quiet corner, a Japanese maple tree stands as a work of art in itself. Its intricately branched silhouette is adorned with finely dissected leaves, each one a masterpiece of intricate patterns. The leaves range from deep maroon to fiery orange, creating a breathtaking gradient of colors throughout the tree. As the sun casts its warm glow, the Japanese maple seems to come alive with a mesmerizing dance of light and shadow. ” analysis the function parameter and call the function to add more trees that fits the text description.  
    Solution: We can identify three different trees (birch trees, dogwood trees, Japanese maple tree) in the scene from the text description: “a cluster of birch trees graces”, “Positioned near a pathway, a dogwood tree…” and “a Japanese maple tree stands…”. From “a cluster of”, we know the tree density is not low, let’s set it to 0.2. The leaves of a birch tree are typically oval. Or triangular in shape and have serrated edges, let’s set the leaf type to ‘leaf’ to allow further customlize adjustment. 
    No fruit for the birch tree in the description. The scene is likely in autumn, the density of leaf_density should be not be too high, let’s set it to 0.02.  
    For the dogwood tree, the leaf are oval or elliptical in shape, with smooth edges and prominent veins, let’s set the leaf type to ‘leaf’ to allow further customlize adjustment. 
    From the description “Clusters of small, vibrant berries cling to its branches”,there are berries in the tree. To choose from the list [’apple’, ’blackberry’, ’coconutgreen’, ’durian’, ’starfruit’, ’strawberry’ ,’None’]. we can choose fruit_type as ‘blackberry’. The scene is likely in autumn, the density of leaf_density should be not be too high, let’s set it to 0.02. For the Japanese maple tree, we can directly set the leaf type to ‘leaf_maple’. No fruit for the  maple tree in autumn. The scene is likely in autumn, the leaf density for the maple tree should be high, let’s set it to 0.1.  We can call the function tree times to create tree different trees:
    ```python add_trees(scene, density = 0.01, leaf_type = ‘leaf’, fruit_type=’None’, leaf_density =  0.02 ) # create birch trees 
      add_trees(scene, density = 0.01, leaf_type = ‘leaf’, fruit_type=’blackberry’, leaf_density =  0.02 ) # create dogwood trees 
      add_trees(scene,density=0.01,leaf_type=‘leaf_maple’, fruit_type=‘None’,leaf_density = 0.03) # create japanese maple trees  
    ```  
    """

    return usage,info, code,document,example


def tree_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    # Define the regex pattern to match the content between '' and 'ddere'
    pattern = r'add_trees\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'

    counter = 0
    # set default dictionary to handle wrong response
    dictionary = {"tree_params": []}
    tree_params = []

    try:
        #Loop through the matches and print each one
        for match in matches:
            counter += 1
            extracted_text = match.group(1)
            pattern = r',(?![^\[]*\])'  # Define the regex pattern to match commas outside of square brackets
            result = re.split(pattern, extracted_text)  # Use re.split to split the string based on the pattern
            parameters = [s.strip() for s in result]  # Trim spaces from the resulting strings
            parameter_length = len(parameters)
            if (parameter_length != 6):
                state = 'Parameter length not correct'
                continue

            density = float(parameters[1].split('=')[1])
            distance_min = float(parameters[2].split('=')[1])
            leaf_type = str(parameters[3].split('=')[1]).replace('\'','').replace('\"','')
            fruit_type = str(parameters[4].split('=')[1]).replace('\'','').replace('\"','')
            leaf_density = float(parameters[5].split('=')[1])

            leaf_list = ["leaf", "leaf_broadleaf", "leaf_ginko", "leaf_maple", "flower", "None"]
            fruit_list = ["apple", "blackberry", "coconutgreen", "durian", "starfruit", "strawberry", "None"]

            def is_sublist(a, b):
                return set(b).issubset(set(a))

            check_data_type = isinstance(density, float) and isinstance(distance_min, float) \
                              and isinstance(leaf_type, str) and isinstance(fruit_type, str) \
                              and isinstance(leaf_density, float) and is_sublist(leaf_list, [leaf_type]) and \
                              is_sublist(fruit_list, [fruit_type])


            if (not check_data_type):
                state = 'Data type error'
                return state, None

            tree_param = {
                'density': density,
                'distance_min': distance_min,
                'control':{
                'leaf_type': leaf_type,
                'fruit_type': fruit_type,
                'leaf_density': leaf_density}
            }

            tree_params.append(tree_param)

        dictionary["tree_params"] = tree_params

    except:
        state = 'No match found'
        return state, None

    if (counter == 0):
        state = 'No match found'
        return state, None

    return state, dictionary

if __name__ == "__main__":
    usage,info, code,document,example = get_tree_modeling_documents()
    text_description = "Among the ancient residents of the forest, the mighty sequoia trees stand with a dignified presence. Their colossal trunks, wide as a house, seem to touch the very essence of time itself. Bark, thick and fibrous, carries the weight of centuries, and the branches high above host a symphony of needles that filter the sunlight into a gentle, mystical glow. These giants create a world within a world, a sanctuary of awe and wonder. The beech trees in this forest contribute to the enchanting atmosphere. Their smooth, gray trunks rise with a sense of elegance, forming a natural colonnade beneath the leafy canopy. Leaves, waxy and dark green, flutter like delicate banners in the wind. As sunlight trickles through, it casts intricate shadows on the forest floor, inviting the wanderer to pause and reflect.The sturdy fir trees reach towards the sky, their branches adorned with needles that exude an invigorating fragrance. In the embrace of their boughs, the forest's secret stories unfold. The needles, deep green and aromatic, catch raindrops and sunlight alike, creating a serene melody of trickling water and dappling light that echoes through the tranquil woods. Tucked within the vibrant mosaic of foliage, the redwood trees rise with grace and grandeur. Their reddish-brown bark, soft to the touch, contrasts against the verdant surroundings. The needles, soft and feathery, embrace the sunlight in an evergreen embrace, casting a verdant hue across the forest floor. Among these redwoods, the sense of time takes on new dimensions, a reminder of nature's enduring legacy. "

    augmented_text, _ =  conceptulization_call(text_description,info,max_tokens=500)
    print('augmented text',augmented_text)
    text_description = text_description + augmented_text # append augmented_text to the initial text description.

    text,_ = modeling_function_call(text_description=text_description,function_description=usage,function_document=document,function=code,example=example)

    print(text)
    print('*' * 8)
    state, dictionary = tree_parser(text)
    print('state',state,'modeling dictionary:', dictionary)
    print('*' * 8)
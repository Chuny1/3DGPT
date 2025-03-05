import sys
sys.path.append('../LLM')
from agents.modeling_agent import modeling_function_call
from agents.conceptulization_agent import conceptulization_call
from documents import *
import openai
import json
# put your api key here.
openai.api_key = ''
# default scene dictionary
def get_default_scene_dict():
    bush_control_dict = {
        'shrub_shape': 2, # range(1,2)
        'max_distance' : 20,  # range = [20, 30, 40, 50, 60, 70]
        'n_twig': 2,
        'n_leaf':50,
        'leaf_type': 'flower', # ['leaf', 'leaf_v2', 'leaf_broadleaf', 'leaf_ginko', 'leaf_maple','flower', 'berry',None]
    }
    control_params = {"surface":{},
                      'lighting':{
                          "dust_density":0.01,
                          "air_density":1,
                          "strength": 0.18, # (0.18,0.22)
                          "sun_intensity": 0.8, # (0.8,1)
                          "sun_elevation":10,
                          "ozone_density":1,
                      },
                      'terrain':{
                          'ground':True, # ground collection
                          'landtiles':False, # mountain_collection
                          'warped_rocks':False, # mountain_collection
                          'voronoi_rocks': False,# mountain_collection
                          'voronoi_grains':False, # always false, nothing created
                          'upsidedown_mountains':False, # mountain_collection
                          'volcanos': False, #
                          'ground_ice': False,
                          'waterbody': True, # liquid_collection
                      }, # refer to config/base_surface_registry.gin to find the default setting.
                      'n_tree_species': 1,  # define the number of tree species
                      'n_bush_species':1,
                      'bush_control':bush_control_dict,
                      'bush_density': 0.01, # (0.015, 0.2)
                      'n_boulder_species' : 2,
                      'n_cactus_species':0,
                      'cactus_density':0.001,
                      'boulder_density':0.02,
                      'glow_rock_density':0,
                      'add_ground_twigs':False,
                      'has_leaf_particles': False,
                      'has_rain_particles': False,
                      'has_dust_particles': False,
                      'has_marine_snow_particles': False,
                      'has_snow_particles': False,
                      'add_rock':False,
                      'add_grass': True,
                      'add_monocots': False,
                      'add_ferns': True,
                      'add_flowers': True,
                      'add_pine_needle':False,
                      'add_decorative_plants':True,
                      'add_chopped_trees': True,
                      'add_ground_leaves': False,
                      'populate':{
                          'slime_mold_on_trees_per_instance_chance':0,
                          'lichen_on_trees_per_instance_chance':0,
                          'ivy_on_trees_per_instance_chance':0,
                          'mushroom_on_trees_per_instance_chance':0,
                          'moss_on_trees_per_instance_chance':0,
                          'slime_mold_on_boulders_per_instance_chance': 0,
                          'lichen_on_boulders_per_instance_chance': 0,
                          'ivy_on_boulders_per_instance_chance': 0,
                          'mushroom_on_boulders_per_instance_chance': 0,
                          'moss_on_boulders_per_instance_chance': 0,
                      }
                      }
    return control_params


def agent_call(text_description,function,function_parser,vis_text = True, maximum_try = 15):
    """
    One agent call consists conceputlization call and modeling call
    """
    # get required information
    usage, info, code, document, example = function
    augmented_text, _ = conceptulization_call(text_description, info)
    text = text_description + augmented_text  # append augmented_text to the initial text description.

    for i in range(maximum_try):
        modeling_text, _ = modeling_function_call(text_description=text, function_description=usage,
                                                 function_document=document, function=code,
                                                 example=example)
        state, dictionary = function_parser(modeling_text)

        if("Pass" in state):
            if (vis_text):
                print('response is:', modeling_text)
                print('dictionary:', dictionary)
                print('*' * 10)
            break

    return state,dictionary

def parser(default_dictionary,text_description,maximum_try = 15,vis_text = True):

    # get terrain_base
    state,dictionary_terrain  = agent_call(text_description, get_terrain_modelling_documents(),
                                           terrain_modelling_parser, vis_text=vis_text, maximum_try=maximum_try)
    if("Pass" in state):
        default_dictionary["terrain"] = dictionary_terrain["terrain"]

    # get terrain surface
    state, dictionary_terrain_surface = agent_call(text_description,get_terrain_surface_documents(),
                                           terrain_surface_parser,vis_text=vis_text,maximum_try=maximum_try)
    if("Pass" in state):
        default_dictionary["surface"] = dictionary_terrain_surface

    # get sky modelling
    state, dictionary_sky = agent_call(text_description,get_sky_modelling_documents(),
                                           sky_modelling_parser,vis_text=vis_text,maximum_try=maximum_try)
    if("Pass" in state):
        default_dictionary["lighting"] = dictionary_sky["lighting"]

    # add elements
    state, dictionary_elements = agent_call(text_description,get_add_elements_documents(),
                                           add_elements_parser,vis_text=vis_text,maximum_try=maximum_try)
    if ("Pass" in state):
        for key in dictionary_elements:
            default_dictionary[key] = dictionary_elements[key]
        if(dictionary_elements.get("add_snow")):
            default_dictionary["surface"]["ground_collection"] = "snow"
            default_dictionary["surface"]["mountain_collection"] = "snow"

    # add particle
    state, dictionary_particle = agent_call(text_description,get_add_floating_particle_documents(),
                                           add_floating_particle_parser,vis_text=vis_text,maximum_try=maximum_try)

    if ("Pass" in state):
        for key in dictionary_particle:
            default_dictionary[key] = dictionary_particle[key]

    # add trees
    state, dictionary_trees = agent_call(text_description,get_tree_modeling_documents(),
                                           tree_parser,vis_text=vis_text,maximum_try=maximum_try)

    if ("Pass" in state):
        default_dictionary["tree_species_params"] = dictionary_trees["tree_params"]


    print("*"*10)
    print(default_dictionary)
    print("*" * 10)

    return default_dictionary



scene_text_list = [
"A serene sunrise over a calm, mist-covered lake in the mountains.",
# "Lush, emerald-green meadows stretching as far as the eye can see, dotted with wildflowers.",
# "A cascading waterfall surrounded by dense, ancient forest.",
# "A tranquil beach with crystal-clear turquoise waters and white sandy shores.",
# "A vibrant, colorful sunset painting the sky with shades of orange and pink over a coastal cliff.",
# "A dense, foggy forest with towering trees and moss-covered rocks.",
# "A field of sunflowers swaying gently in the breeze under a bright blue sky.",
# "A snow-capped mountain range reflecting in a pristine alpine lake.",
# "A desert oasis with palm trees, a shimmering pool, and sand dunes in the distance.",
# "A peaceful countryside scene with rolling hills and a rustic farmhouse.",
]


for i,text in enumerate(scene_text_list):
    print("#"*10,i)
    dictionary = get_default_scene_dict()
    text_description = text
    print('text input:',text_description)
    dictionary = parser(dictionary,text_description,maximum_try=15)



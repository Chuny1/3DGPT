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

def get_sky_modelling_documents():
    usage = """
    modelling the sky appearance using nishita method
    """

    code = """
    import nishita_sky_modelling
    def sky_texture_node(sun_intensity,sun_elevation,sun_rotation,air_density,dust_density, ozone,cloud_density):
        texture = nishita_sky_modelling(sun_intensity,sun_elevation,sun_rotation,air_density,dust_density, ozone,cloud_density)
        return 
    """

    info = """
    possible daytime, sun intensity, air condition, sky appearance, cloud density. 
    """

    document = """
    input: 
    sun_intensity: Multiplier for sun disc lighting. (choose from ‘low’,’median’,’high’)
    sun_elevation: Rotation of the sun from the horizon (in degrees). (0:sunset,sun rising, 90:daytime,-10:night)
    sun_rotation: Rotation of the sun around the zenith (in degrees).
    air_density: density of air molecules. (0 no air, 1 clear day atmosphere, 2 highly polluted day)
    dust_density: density of dust and water droplets. (0 no dust, 1 clear day atmosphere, 5 city like atmosphere, 10 hazy day) 
    ozone: density of ozone molecules; useful to make the sky appear bluer. (0 no ozone, 1 clear day atmosphere, 2 city like atmosphere). higher value for bluer sky.
    cloud_density: density of the cloud, varying from 0 to 0.04. (0.01 very thick cloud, 0.04 very heavy cloud)
    output:  texture color output.
    """

    example = """
    Question:  given the text description of the scene: “The river, reflecting the clear blue of the sky, glistened like a silver ribbon as it wound its way through the lush valley, its tranquil waters whispering secrets to the ancient trees.” analysis the function parameter and call the function to generate the sky. 
    Solution: From the description of “clear blue sky”, the sun_intensiy can not be low, let's set it as "median";As the sky is blue, the sun_elevation cannot be very low (not morning/sunset/evening), we can set it to 50.
     The sun rotation does not affect the sky appearance, let's set it to 0. To fit the description of "clear blue". The air_density and dust_density should be low, we can set air_density to 1, dust_density to 0.  To make the sky more blue, we can increase the ozone value, let’s make it as 2. 
     As the sky is very blue and clear, the cloud density should be low, we can set it as 0.0005. We can model the sky by calling the following function:
    ```python sky_texture_node(sun_intensity = ‘median’,sun_elevation=50,sun_rotation=0,air_density=1,dust_density=0, ozone=2,cloud_density=0.0005)```
    """
    return usage, info, code, document, example


def query_match(pattern,input_text):
    # Use re.search to find the first match of the pattern in the input text
    match = re.search(pattern, input_text)

    # Check if a match is found
    if match:
        extracted_string = match.group(1)
        return extracted_string
    else:
        return ''

def query_digits(pattern = r'\d+',input_text = ''):
    # Use re.findall to find all matches of the pattern in the input string
    matches = re.findall(pattern, input_text)
    # Convert the matched strings to integers
    numbers = [int(match) for match in matches]
    return numbers

def sky_modelling_parser(text):
    """
    get the agent response return the response state and extracted information
    """
    pattern = r'sky_texture_node\((.*?)\)'
    # Use re.finditer to find all matches
    matches = re.finditer(pattern, text)
    state = 'Pass'
    counter = 0
    # set default dictionary to handle wrong response

    dictionary = {
        "lighting": {
            "dust_density": 0.01,
            "air_density": 2,
            "strength": 0.18,  # (0.18,0.22)
            "sun_intensity": 0.8,  # (0.8,1)
            "sun_elevation": 20,
            "sun_rotation":0,
            "ozone_density": 1,
            "cloud_density":0.01, # (0.01,0.04)
        }
    }
    try:
        # Loop through the matches and print each one
        for match in matches:
            counter += 1
            extracted_text = match.group(1)
            parameters = extracted_text.split(',')
            parameter_length = len(parameters)
            if(parameter_length!=7):
                state = 'Parameter length not correct'
                break

            sun_intensity = query_match(r"'(.*?)'", parameters[0]).replace(' ','')
            sun_intensity_map = {'low':0.6, 'median':0.8, 'high':0.95}
            sun_intensity = sun_intensity_map[sun_intensity]
            sun_elevation = float(parameters[1].split('=')[1])
            sun_elevation_sign = '-' in parameters[1]
            sun_elevation = -sun_elevation if sun_elevation_sign else sun_elevation
            sun_rotation = float(parameters[2].split('=')[1])
            air_density = float(parameters[3].split('=')[1])
            dust_density = float(parameters[4].split('=')[1])
            ozone = float(parameters[5].split('=')[1])
            cloud_density = float(parameters[6].split('=')[1])

            dictionary["lighting"]["dust_density"] = dust_density
            dictionary["lighting"]["sun_intensity"] = sun_intensity
            dictionary["lighting"]["sun_elevation"] = sun_elevation
            dictionary["lighting"]["sun_rotation"] = sun_rotation
            dictionary["lighting"]["air_density"] = air_density
            dictionary["lighting"]["ozone_density"] = ozone
            dictionary["lighting"]["cloud_density"] = cloud_density

            data_type_check = isinstance(dust_density, (int, float)) and \
                              isinstance(sun_intensity, (int, float)) and \
                              isinstance(sun_elevation, (int, float)) and \
                              isinstance(sun_rotation, (int, float)) and \
                              isinstance(air_density, (int, float)) and \
                              isinstance(ozone, (int, float)) and \
                              isinstance(cloud_density, (int, float))

            if(not data_type_check):
                state = 'Data type error'
                return state, None

            break
    except:
        state = 'No match found'
        return state, None

    if(counter==0):
        state = 'No match found'
        return state, None

    return state, dictionary

if __name__ == "__main__":

    usage,info, code,document,example = get_sky_modelling_documents()
    text_description = "The beach, a tapestry of sun-kissed sand, stretched out for miles, caressed by the gentle waves that rhythmically lapped against the shore, as seagulls soared and dove in graceful arcs above the sparkling sea."

    augmented_text, _ =  conceptulization_call(text_description,info)
    print('augmented text',augmented_text)
    text_description = text_description + augmented_text # append augmented_text to the initial text description.


    text,_ = modeling_function_call(text_description=text_description,function_description=usage,function_document=document,function=code,example=example)
    state, dictionary = sky_modelling_parser(text)
    print(text)
    print('state',state)
    print('dictionary',dictionary)

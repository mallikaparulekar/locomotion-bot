import xml.etree.ElementTree as ET
import numpy as np

def extract_link_masses(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    mass_dict = {}
    for link in root.findall('link'):
        link_name = link.get('name')
        mass_element = link.find('./inertial/mass')
        if mass_element is not None:
            mass_value = float(mass_element.get('value'))
            mass_dict[link_name] = mass_value
        else:
            mass_dict[link_name] = None  # No mass specified for this link
    
    return mass_dict




def get_three_largest_masses(link_masses):
    # Sort the link masses in descending order
    sorted_link_masses = sorted(link_masses.items(), key=lambda x: x[1], reverse=True)
    # Get the first three elements from the sorted list
    three_largest_masses = sorted_link_masses[:3]
    return three_largest_masses


def add_gaussian_noise(mass_dict, keys_to_modify, mean=0, std_dev=0.01):
    noisy_mass_dict = mass_dict.copy()
    for key in keys_to_modify:
        if key in noisy_mass_dict and noisy_mass_dict[key] is not None:
            noisy_mass_dict[key] += np.random.normal(mean, std_dev)
    return noisy_mass_dict



def update_urdf_masses(original_urdf_path, new_urdf_path, updated_masses):
    tree = ET.parse(original_urdf_path)
    root = tree.getroot()
    
    for link in root.findall('link'):
        link_name = link.get('name')
        if link_name in updated_masses:
            mass_element = link.find('./inertial/mass')
            if mass_element is not None:
                mass_element.set('value', str(updated_masses[link_name]))
    
    tree.write(new_urdf_path)




def create_multiple_urdfs(original_urdf, new_urdf_prefix, num_urdfs, mean=0, std_dev=0.01):
    link_masses = extract_link_masses(original_urdf)
    three_largest_masses = get_three_largest_masses(link_masses)
    keys_to_add_noise = [link_name for link_name, _ in three_largest_masses]
    
    for i in range(num_urdfs):
        noisy_link_masses = add_gaussian_noise(link_masses, keys_to_add_noise, mean, std_dev)
        new_urdf_file = f"{new_urdf_prefix}_{i}.urdf"
        update_urdf_masses(original_urdf, new_urdf_file, noisy_link_masses)


create_multiple_urdfs(
    "/Users/mallikaparulekar/Desktop/CS/CS234/Project/genesis_playground/genesis_playground/resources/zbot/robot_fixed.urdf",
    "/Users/mallikaparulekar/Desktop/CS/CS234/Project/genesis_playground/genesis_playground/resources/zbot/robot_fixed_noisy",
    20,
    mean=0.05,
    std_dev=0
)


# Example usage:
# urdf_file = "/Users/mallikaparulekar/Desktop/CS/CS234/Project/genesis_playground/genesis_playground/resources/zbot/robot_fixed.urdf"  
# link_masses = extract_link_masses(urdf_file)
# three_largest_masses = get_three_largest_masses(link_masses)

# keys_to_add_noise = [link_name for link_name, _ in three_largest_masses]
# noisy_link_masses = add_gaussian_noise(link_masses, keys_to_add_noise, mean=0, std_dev=0.05)

# new_urdf_file = "/Users/mallikaparulekar/Desktop/CS/CS234/Project/genesis_playground/genesis_playground/resources/zbot/robot_fixed_noisy.urdf"
# update_urdf_masses(urdf_file, new_urdf_file, noisy_link_masses)
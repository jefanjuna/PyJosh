import yaml

with open('test_config.yml', 'r') as file:
    data = yaml.safe_load(file)

print(data)
status = 'test'
id_input_file = f'../{status}_audio_id.txt'
domain_input_file = f'../{status}_fake_do.txt'

id_list = []
domain_list = []

with open(id_input_file, mode='r') as f1:
	id_lines = f1.readlines()
	for id_line in id_lines:
		id_line = id_line.strip()
		id_list.append(id_line)

with open(domain_input_file, mode='r') as f2:
	domain_lines = f2.readlines()
	for domain_line in domain_lines:
		domain_line = domain_line.strip()
		domain_list.append(domain_line)

for i in range(len(id_list)):
	print(f'{id_list[i]},"[\'{domain_list[i]}\']"')

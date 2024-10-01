status = 'test'
input_file = f'{status}_output.txt'

with open(input_file, mode='r') as f:
	lines = f.readlines()
	for line in lines:
		line = line.strip()
		new_line = f'이 데이터셋은 SiTEC Dict01로, 한국어 문장을 읽는 400명의 다양한 연령대 화자들로 구성되어 있습니다. 녹음은 다양한 거리(3m 이상)에서 진행되었습니다. 지금부터 이 데이터셋을 사용해 음성 인식 태스크 훈련을 시작할 예정이며, 각 음성 파일에 거리 정보를 추가할 것입니다. 거리 정보를 고려하여 모델이 정확하게 학습되도록 해주세요. 이 음성의 녹음 거리는 {line}m입니다.'
		print(new_line)

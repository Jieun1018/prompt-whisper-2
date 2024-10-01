import csv

# 입력 파일과 출력 파일 경로 설정
status = 'train'
input_file = f'{status}_fake_domain_1.csv'
output_file = f'{status}_audio_id.csv'

# 출력 파일을 열고 쓰기 모드로 설정
with open(output_file, mode='w', newline='') as out_csv:
    writer = csv.writer(out_csv)
    writer.writerow(['audio_id'])  # 헤더 작성
    #writer.writerow(['audio_id', 'distance_m'])  # 헤더 작성
    
    # 입력 파일 읽기
    with open(input_file, mode='r') as in_csv:
        reader = csv.DictReader(in_csv)
        
        # 각 행에 대해 처리
        for row in reader:
            audio_id = row['audio_id']
            #domain = row['domain']
            
            # 'distance' 뒤의 미터 값을 추출
            '''
			start = domain.find('distance') + len('distance ')
            end = domain.find('m', start)
            distance_m = domain[start:end].strip()
            '''
            # 새로운 CSV 파일에 작성
            #writer.writerow([audio_id, distance_m])
            writer.writerow([audio_id])

print(f"Distance values have been extracted and saved to {output_file}.")


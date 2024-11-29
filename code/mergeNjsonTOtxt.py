import json

def merge_json_files(directory, output_file):
    json_files = glob.glob(os.path.join(directory, '*.json'))
    merged_data = []

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            file_data = json.load(file)  
            merged_data.append(file_data)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, ensure_ascii=False, indent=2)

merge_json_files('./train', 'train_1.json')
merge_json_files('./test', 'test_1.json')

with open('train_1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

essays = [entry['essay_txt'] for entry in data if isinstance(entry, dict)]
with open('train_2.txt', 'w', encoding='utf-8') as txt_file:
    for essay in essays:
        txt_file.write(essay + "\n")

with open('test_1.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

essays = [entry['essay_txt'] for entry in data if isinstance(entry, dict)]
with open('test_2.txt', 'w', encoding='utf-8') as txt_file:
    for essay in essays:
        txt_file.write(essay + "\n")
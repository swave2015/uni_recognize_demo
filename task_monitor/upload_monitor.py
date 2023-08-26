import os
import time
import yaml
import threading
import shutil
import datetime
import uuid
import zipfile

class UploadMonitor:
    
    def __init__(self, recognize_queue, retrival_queue, config_file="./config/config.yaml", duration=5, scan_interval=10):
        self.config_file = config_file
        self.duration = duration
        self.scan_interval = scan_interval
        self.tasks = self.get_tasks_from_config()
        self.recognize_queue = recognize_queue
        self.retrival_queue = retrival_queue

    # def __init__(self, config_file="/data/xcao/code/uni_recognize_demo/config/config.yaml", duration=5, scan_interval=10):
    #     self.config_file = config_file
    #     self.duration = duration
    #     self.scan_interval = scan_interval
    #     self.tasks = self.get_tasks_from_config()
        
    def get_tasks_from_config(self):
        with open(self.config_file, 'r') as file:
            data = yaml.safe_load(file)
        print('config_data: ', data)
        return data['tasks']
    
    def has_file_upload_completed(self, filepath):
        initial_mod_time = os.path.getmtime(filepath)
        time.sleep(self.duration)
        return os.path.getmtime(filepath) == initial_mod_time
    
    def get_recognize_prompts(self, directory):
        def get_prompt(filename):
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                return [line.strip() for line in lines if line.strip() != '']
            return None

        targets = ['target_actions']
        prompt_dict = {}

        for target in targets:
            prompt = get_prompt(f'{target}.txt')
            if prompt:
                prompt_dict[target] = prompt

        return prompt_dict
    
    def get_search_prompts(self, directory):
        def get_prompt(filename):
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                return [line.strip() for line in lines if line.strip() != '']
            return None

        targets = ['search_texts']
        prompt_dict = {}

        for target in targets:
            prompt = get_prompt(f'{target}.txt')
            if prompt:
                prompt_dict[target] = prompt

        return prompt_dict

    def downstream_task(self, task, ori_filename, zip_filepath):
        # Replace with your actual downstream task
        print(f"Starting downstream task for filename: {zip_filepath}")
        dir_path = os.path.dirname(zip_filepath)
        filename = os.path.basename(zip_filepath)
        filename_without_ext =  os.path.splitext(filename)[0]
        extract_dir = os.path.join(dir_path, filename_without_ext)
        now = datetime.datetime.now()
        time_string = now.strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(task['done_directory'], ori_filename + '_' + time_string)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(zip_filepath)
        if task['type'] == 'recognize':
            prompt_dict = self.get_recognize_prompts(extract_dir)
            print(f"prompt_len: {len(prompt_dict)}")
            if len(prompt_dict) == 0:
                print(f"No prompts in zip package, delete all of them {extract_dir}")
                shutil.rmtree(extract_dir)
            else:
                print(f"prompt_dict: {prompt_dict}")
                item = (extract_dir, prompt_dict, save_dir)
                self.recognize_queue.put(item)
        if task['type'] == 'retrieval':
            prompt_dict = self.get_search_prompts(extract_dir)
            print(f"prompt_len: {len(prompt_dict)}")
            if len(prompt_dict) == 0:
                print(f"No prompts in zip package, delete all of them {extract_dir}")
                shutil.rmtree(extract_dir)
            else:
                print(f"prompt_dict: {prompt_dict}")
                item = (extract_dir, prompt_dict, save_dir)
                self.retrival_queue.put(item)

    def wait_for_all_uploads_to_complete(self, task):
        while True:
            if not os.listdir(task['upload_directory']):
                time.sleep(self.scan_interval)
                continue
            for filename in os.listdir(task['upload_directory']):
                filepath = os.path.join(task['upload_directory'], filename)

            # Check and delete files that are not .zip
            if os.path.isfile(filepath):
                _, ext = os.path.splitext(filename)
                if ext != '.zip':
                    print(f"Deleting non-zip file: {filepath}")
                    os.remove(filepath)
                    continue
                
                print(f"File {filepath} found...")
                if not self.has_file_upload_completed(filepath):
                    print(f"File {filepath} is still being written to...")
                else:
                    now = datetime.datetime.now()
                    time_string = now.strftime('%Y%m%d_%H%M%S')
                    unique_id = uuid.uuid4()
                    target_path = os.path.join(task['tmp_directory'], str(unique_id) + '_' + time_string + '_' + os.path.basename(filepath))
                    shutil.move(filepath, target_path)
                    # shutil.copy(filepath, target_path)
                    print(f"Move file || src: {filepath} || target: {target_path}")
                    ori_filename =  os.path.basename(filepath)
                    ori_filename_without_ext = os.path.splitext(ori_filename)[0]
                    self.downstream_task(task, ori_filename_without_ext, target_path)

            # Check and delete sub-directories
            elif os.path.isdir(filepath):
                print(f"Deleting sub-directory: {filepath}")
                shutil.rmtree(filepath)

            time.sleep(self.scan_interval)

    def start_monitoring(self):
        for task in self.tasks:
            thread = threading.Thread(target=self.wait_for_all_uploads_to_complete, args=(task,))
            thread.start()
            print(f"start task monitor: {task['type']}")
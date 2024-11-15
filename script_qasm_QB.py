import time
from pathlib import Path
import argparse
from qctl.core.cloud_platform_client import CloudPlatformClient
from itertools import islice
import nltk
import re
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit
from qiskit import qasm2

from torch.nn import (
    Module,
    Linear
)
import torch.nn.functional as F

from qiskit_machine_learning.connectors import TorchConnector


nltk.download('punkt') # делит текст на список предложений
nltk.download('wordnet') # проводит лемматизацию
lemmatize = nltk.WordNetLemmatizer()
nltk.download('punkt_tab')

CLOUD_PLATFORM_URL = "https://cloudos.qboard.tech"
WORKSPACE_ID = "workspace-6929b2f045c54eaeb234adde1ccb62c8"  # workspace should be already exists
IMAGE_NAME = "cloud-platform-quantum-modules-qboard"  # use correct image name

class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.fc1 = Linear(dt_to_pred.shape[1], 64)
        self.fc2 = Linear(64, 2)  # 2-dimensional input to QNN
        self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
        # uniformly at random from interval [-1,1].
        self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)  # apply QNN
        x = self.fc3(x)
        return torch.cat((x, 1 - x), -1)

def print_status(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} {msg}")


def process_file(input_file_path: Path, output_file_path: Path, use_gpu: bool, user_data_path: Path):
    """Processes the input file and writes the results to the output file."""

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    if output_file_path and not output_file_path.parent.exists():  # Check if parent directory exists
        raise FileNotFoundError(f"Output directory does not exist: {output_file_path.parent}")

    if not user_data_path.exists():  # Check if parent directory exists
        raise FileNotFoundError(f"User data file does not exist: {output_file_path.parent}")

    try:
        with open(user_data_path, mode="r", encoding="utf-8") as f:
            username, password = [line.strip() for line in islice(f, 2)]
            print_status(f"{username=}")
    except OSError:
        print_status(f"ERROR: Could not open/read file: {user_data_path.name}")

    run_cloudos_task(input_file_path, output_file_path, use_gpu, username, password)


def run_cloudos_task(input_file_path: Path, output_file_path: Path, use_gpu: bool, username: str, password: str):
    # auth process
    client = CloudPlatformClient(cloud_platform_url=CLOUD_PLATFORM_URL)

    client.login(username=username, password=password)
    input_filename = input_file_path.name
    if output_file_path:
        output_filename = output_file_path.name
    else:
        output_filename = Path("output.json")
        
    # putting file on the server
    print_status(f"Loading file {input_file_path} on the server...")
    client.put_file(source_path=input_file_path, dest_path=input_filename, workspace_id=WORKSPACE_ID)
    print_status(f"File {input_filename} loaded on the server.")

    # creating ideem process
    print_status("Creating process...")

    default_args = [
            f"/workspace/{input_filename}",
            "--shots",
            "1000",
            "--output",
            f"/workspace/{output_filename}",
            "--verbose",
            "1",
            "--txt",
        ]

    if use_gpu:
        default_args.append("--gpu")

    process = client.create_process(
        workspace_id=WORKSPACE_ID,
        image=IMAGE_NAME,
        name="run ideem",
        cpu="4",
        ram="4",
        gpu=int(use_gpu == True),
        command="ideem",
        args=default_args,
    )
    print_status("Process created.")

    # Waiting for process to complete
    print_status("Waiting for process to complete...")
    while process["status"] != "COMPLETED":
        if process["status"] in ["CREATED", "SUBMITTED", "RUNNING", "COMPLETED"]:
            time.sleep(1)
        else:
            msg = client.get_process_output(process["id"])
            print_status("The process was not completed normally")
            raise ValueError(msg)
        process = client.describe_process(process["id"])
    print_status("Process completed.")

    # print process output
    process_output = client.get_process_output(process["id"])
    print_status(f"Process_output: {process_output}")

    # load file from server to local dir
    if output_file_path:
        print_status(f"Getting output file {output_filename} from the server...")
        client.get_file(source_path=output_filename, dest_path=output_file_path, workspace_id=WORKSPACE_ID)
        print_status(f"File saved locally to {output_file_path}.")


def preprocess_text(phrase):
    phrase_clear = []
    for i in phrase:
        #удаляем неалфавитные символы
        text = re.sub("[^а-яёА-ЯЁ]"," ",i.lower())
        # токенизируем слова
        text = nltk.word_tokenize(text,language = "russian")
        # лемматирзируем слова
        lemm_text = []
        for word in text:
            # if word not in set(stopwords.words("russian")):
            #     lemm_text.append(lemmatize.lemmatize(word))
            if word not in set(stopwords):
                lemm_text.append(lemmatize.lemmatize(word))
        #print(text)
        # соединяем слова
        lemm_text = " ".join(lemm_text)
        phrase_clear.append(lemm_text)
    return(phrase_clear)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file and write the results to another file.", allow_abbrev=False)
    parser.add_argument("--input-file", "-i", type=Path, required=True, help="Path to the input file. Should be in edgelist format!")
    parser.add_argument("--output-file", "-o", type=Path, required=False, help="Path to the output file, json format")
    parser.add_argument("--user-data-file", "-u", type=Path, required=False, help="Path to the user data file.", default="./USER_DATA.txt")
    parser.add_argument("--gpu", action='store_true', help="Use GPU.")
    parser.add_argument("--run-id", type=int, required=False, help="Run id for parallel launch.", default=-1)

    args = parser.parse_args()

    if args.run_id != -1:
        output_file_path = args.output_file.with_name(f"{args.run_id}_{args.output_file.name}") 
    else:
        output_file_path = args.output_file
    #Загрузим список стоп-слов для обработки новых отзывов
    with open("stopWords.txt") as file:
        data = [line.strip() for line in file]
    stopwords = []
    for word in data:
        word = word.replace(',', "")
        word = word.replace('\'', "")
        stopwords.append(word)
    #загрузим обученную модель векторизации
    with open('bOfW.pkl', 'rb') as f:
        cntVec, tfidfVec = pickle.load(f)
    #тестовый придуманный отзыв
    review = 'смартфон не понравился, батарея слабая, экран сломался'
    dt_to_pred = preprocess_text([review])
    dt_to_pred = tfidfVec.transform(dt_to_pred)
    dt_to_pred = dt_to_pred.toarray()
    dt_to_pred = torch.tensor(dt_to_pred, dtype=torch.float32)
    #загрузим обученную нейросеть
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    #определим параметры вариационного слоя квантовой нейросети, полученные в результате обучения
    theta0, theta1, theta2, theta3 = model.state_dict()['qnn.weight'].numpy()
    x = F.relu(model.fc1(dt_to_pred))
    x = model.fc2(x)
    #определим входные параметры блока ZZFeatureMap для данного придуманного отзыва
    x0, x1 = x.detach().numpy()[0]
    #сформируем квантовую цепочку, состоящую из блока ZZFeatureMap и вариационного блока с найденными параметрами
    zzFMQC = QuantumCircuit(2,2)
    zzFMQC.h(0)
    zzFMQC.h(1)
    zzFMQC.p(x0, 0)
    zzFMQC.p(x1, 1)
    zzFMQC.cx(0, 1)
    zzFMQC.p(2*(np.pi-x0)*(np.pi-x1), 1)
    zzFMQC.cx(0, 1)
    zzFMQC.h(0)
    zzFMQC.h(1)
    zzFMQC.p(x0, 0)
    zzFMQC.p(x1, 1)
    zzFMQC.cx(0, 1)
    zzFMQC.p(2*(np.pi-x0)*(np.pi-x1), 1)
    zzFMQC.cx(0, 1)
    zzFMQC.ry(theta0, 0)
    zzFMQC.ry(theta1, 1)
    zzFMQC.cx(0, 1)
    zzFMQC.ry(theta2, 0)
    zzFMQC.ry(theta3, 1)
    zzFMQC.measure(0, 0)
    zzFMQC.measure(1, 1)
    #сформируем для данной цепочки qasm файл
    dumped = qasm2.dumps(zzFMQC)
    with open("QNN.qasm", "w") as file:
        file.write(dumped)
    # полученный qasm файл отправим на эмулятор QBoard (укажем правильное имя файла при вызове данного скрипта)
    try:
        process_file(args.input_file, output_file_path, args.gpu, args.user_data_file)
    except FileNotFoundError as e:
        print_status(f"Error: {e}")
        exit(1)  # Exit with error code
    except Exception as e:
        print_status(f"An unexpected error occurred: {e}")
        exit(1)  # Exit with error code
    
    #загрузим файл с результатами эмуляции и пересчитаем их в вывод выходного слоя квантовой нейросети
    shots = []
    with open("ftest_000_result.txt") as file:
        for line in file:
            shots.append(re.sub("[^0-9:]","",line))
    shotsCl = []
    for sh in shots:
        start = sh.find(':')  
        if start != 0:
            shotsCl.append(int(sh[start+1:start+5]))
    #Вывод выходного слоя определим как ожидаемое наблюдаемой гейта Паули Z^2, где 2 - число кубитов в цепочке. 
    #Собственные векторы состояний 11 и 00 соответствуют собственным числам 1, собственные векторы состояний 
    # 01 и 10 соответствуют собственным числам -1: 
    expVal = (shotsCl[2]+shotsCl[3]-shotsCl[0]-shotsCl[1])/(shotsCl[2]+shotsCl[3]+shotsCl[0]+shotsCl[1])
    print(expVal)
    x = torch.tensor([[expVal]])
    x = model.fc3(x)
    prediction = torch.cat((x, 1 - x), -1).argmax(dim=1, keepdim=True).numpy()[0][0]
    #Выводим результат предсказания
    if prediction == 0:
        print('Отзыв отрицательный')
    else:
        print('Отзыв положительный')

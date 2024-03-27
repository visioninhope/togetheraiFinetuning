DATASET_FILE_PATH = r'D:\TrainingData\ConversionTesting\Converted\20240327T215931\Final\Property_of_Farran_Media_All_Rights_Reserved_20240327T215931_TogetherAI_Format_1000_Tokens.jsonl'

TARGET_MODEL_TO_FINETUNE = 'mistralai/Mistral-7B-Instruct-v0.2'
NUMBER_OF_EPOCHS = 2
NUMBER_OF_CHECKPOINTS = 4
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
FINETUNED_LLM_SUFFIX = 'farranmedia-llm-test-1'

WANDB_API_KEY = open(r'C:\Secret\wandb_api_key.txt').read().strip()
WANDB_URL = 'https://wandb.ai/<username>/together?workspace=user-<username>'
from props import *
import together
together.api_key = open(r'C:\Secret\togetherai_api_key.txt').read().strip()


def togetherai_finetuning_orchestrator():
    # Check the data, essentially valiating the samples within
    print(f'Checking dataset file at {DATASET_FILE_PATH}...')
    resp = together.Files.check(file=DATASET_FILE_PATH)
    print('Checked dataset file (response below).')
    print(resp)
    if resp['is_check_passed']:
        print(f'Uploading dataset file at {DATASET_FILE_PATH}')
        resp = together.Files.upload(file=DATASET_FILE_PATH)
        file_id = resp['id']
        print(f'Uploaded dataset file: file ID = {file_id} (response below).')
        print(resp)
        print(f'Creating fine-tuning job with file ID {file_id}...')
        resp = together.Finetune.create(
            training_file=file_id,
            model=TARGET_MODEL_TO_FINETUNE,
            n_epochs=NUMBER_OF_EPOCHS,
            n_checkpoints=NUMBER_OF_CHECKPOINTS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            suffix=FINETUNED_LLM_SUFFIX,
            wandb_api_key=WANDB_API_KEY,
            confirm_inputs=False
        )
        fine_tune_id = resp['id']
        print(f'Created fine-tuning job with file ID {file_id}: fine-tuning job ID = {fine_tune_id}.')
        print(resp)

    else:
        print('ERROR: files were unsuccessfully checked!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    togetherai_finetuning_orchestrator()

import datetime
import time

from props import *
import together
together.api_key = open(r'C:\Secret\togetherai_api_key.txt').read().strip()


def togetherai_finetuning_orchestrator():
    # Check the data, essentially validating the samples within
    print(f'Checking dataset file at {DATASET_FILE_PATH}...')
    resp = together.Files.check(file=DATASET_FILE_PATH)
    print('Checked dataset file (response below).')
    print(resp)
    if resp['is_check_passed']:
        # Upload the file, and get a file ID
        print(f'Uploading dataset file at {DATASET_FILE_PATH}')
        resp = together.Files.upload(file=DATASET_FILE_PATH)
        file_id = resp['id']
        print(f'Uploaded dataset file: file ID = {file_id} (response below).')
        print(resp)

        # Start the training process, and get the fine-tune ID
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
        print(f'Created fine-tuning job with file ID {file_id}: fine-tuning job ID = {fine_tune_id} (response below).')
        print(resp)

        # This part monitors everything, but you should also use the wandb URL!
        print(f'Monitoring fine-tuning job with ID {fine_tune_id}')
        while not together.Finetune.is_final_model_available(fine_tune_id=fine_tune_id):
            print('---------------------------------------------------------------------------------------------------')
            print('UTC timestamp')
            print(datetime.datetime.utcnow())
            print('retrieve')
            print(together.Finetune.retrieve(fine_tune_id=fine_tune_id))  # retrieves information on finetune event
            print('get_job_status')
            print(together.Finetune.get_job_status(fine_tune_id=fine_tune_id))  # pending, running, completed
            print('get_checkpoints')
            print(together.Finetune.get_checkpoints(fine_tune_id=fine_tune_id))  # list of checkpoints
            time.sleep(30)

        print(f'Monitored fine-tuning job with ID {fine_tune_id} - the job has completed! Good luck!')
    else:
        print('ERROR: files were unsuccessfully checked!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    togetherai_finetuning_orchestrator()

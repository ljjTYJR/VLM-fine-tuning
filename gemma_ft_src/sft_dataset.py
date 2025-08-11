from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch
from gemma_ft_src.constants import (
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
    IM_START_ID,
    PROMPTS
)


# Convert dataset to OAI messages
def format_data(sample, prompt_index):
    prompt = PROMPTS[prompt_index]
    return {"messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },{
                            "type": "image",
                            "image": sample["image"],
                            "resized_height": 360,
                            "resized_width": 640,
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": sample["hddl"]
                        }
                    ],
                },
            ],
        }

def collate_fn_all_inputids(samples, processor):
     # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in samples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in samples]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX
    # Ignore the image token index in the loss computation (model specific)
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = IGNORE_INDEX
    batch["labels"] = labels
    return batch

def collate_fn_ref_ids(samples, processor):
    """ Vectorized batch processing for acceleration """
    # Extract user and assistant messages for all samples
    user_texts = []
    response_texts = []
    image_inputs = []

    for sample in samples:
        user_messages = [msg for msg in sample["messages"] if msg["role"] in ["system", "user"]]
        assistant_messages = [msg for msg in sample["messages"] if msg["role"] == "assistant"]

        user_text = processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=False)
        response_text = assistant_messages[0]["content"][0]["text"] if assistant_messages else ""
        response_text = response_text + '<end_of_turn>' + '\n'

        user_texts.append(user_text)
        response_texts.append(response_text)
        image_inputs.append(process_vision_info(sample["messages"])[0])

    # Batch process user inputs with images
    inputs = processor(text=user_texts, images=image_inputs, return_tensors="pt", padding=False)
    input_ids_batch = inputs.input_ids
    pixel_values_batch = inputs.pixel_values

    # Batch tokenize response texts
    response_tokens = processor.tokenizer(response_texts, return_tensors="pt", padding=True)
    response_ids_batch = response_tokens.input_ids

    # Process each sample and collect tensors
    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_pixel_values = []

    for i in range(len(samples)):
        input_ids = input_ids_batch[i:i+1]
        response_ids = response_ids_batch[i:i+1]
        pixel_values = pixel_values_batch[i:i+1]

        # Concatenate with padding preserved
        all_ids = torch.cat([input_ids[0], response_ids[0]])

        # Create labels: mask input tokens and pad tokens with -100
        labels = torch.cat([
            torch.tensor([IGNORE_INDEX] * input_ids.shape[1]),  # Mask all input tokens
            response_ids[0]  # Keep response tokens (including padding)
        ])
        # Mask pad tokens in labels with -100
        labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (all_ids != processor.tokenizer.pad_token_id).long()

        # Create token type IDs
        token_type_ids = torch.zeros_like(all_ids)
        token_type_ids[all_ids == processor.image_token_id] = 1

        batch_input_ids.append(all_ids)
        batch_labels.append(labels)
        batch_attention_mask.append(attention_mask)
        batch_token_type_ids.append(token_type_ids)
        batch_pixel_values.append(pixel_values[0])

    # Stack all tensors to create batch dimension
    batch = {
        'input_ids': torch.stack(batch_input_ids),
        'labels': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_attention_mask),
        'token_type_ids': torch.stack(batch_token_type_ids),
        'pixel_values': torch.stack(batch_pixel_values)
    }
    return batch

def generate_description(sample, model, processor, prompt_idx):
    # TODO: use new prompt
    prompt = PROMPTS[prompt_idx]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image","image": sample['image'], "resized_height": 360, "resized_width": 640,},
        ]},
    ]
    # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, return_tensors="pt"
    )
    print("The templated text:", text)
    image_inputs, video_inputs = process_vision_info(messages)
    # save the image
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # check whether there is inf or nan in the inputs
    if torch.isnan(inputs.input_ids).any() or torch.isinf(inputs.input_ids).any():
        raise ValueError("Input contains NaN or Inf values.")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

if __name__ == "__main__":
    dataset_id = "shuooru/image-hddl-dataset"

    # dataset = [format_data(sample) for sample in data_loader.dataset]



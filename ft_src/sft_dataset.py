from datasets import load_dataset
from qwen_vl_utils import process_vision_info
import torch
from ft_src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    SYSTEM_MESSAGE,
    IM_START_ID,
    IM_END_ID,
)

old_prompt= """
I want you to create HDDL problem file (similar to pddl file) of the image that I give as input.
An example of an HDDL is this:
(define
        (problem pfile01)
        (:domain  domain_htn)
        (:objects
                plate1 - container
                pear1 - food
                home1 wp1s wp2s - location
                wp1f - location
                robot1 - robot
        )       (:htn
                :parameters ()
                :subtasks (and
                 (task0 (move_object plate1 wp1f))
                 (task1 (move_to_container pear1 plate1))
                )
                :ordering (and
                )
        )

        (:init
                (at plate1 wp1s)
                (at pear1 wp2s)
                (at robot1 home1)
        )
)
Just differentiate between food, container (plate,basket,cup,bowl) and the rest of the object can be listed as items.
For the location of the objects, use simply wp1s, wp2s (for the start) and wp1f, wp2f (for the goal).
For the goal, only food and containers are allowed on the table.
Put food in containers and remove the other object from the tables.
The task you can use are: move_object (to move the objects), move_to_container (to move objects to the container).
To remove the object, use the task (move_object, remote_control, out_location).
To move the objects, use (move_object plate wp1f).
Only output the generated hddl languages.
"""

prompt0= """
I want you to create HDDL problem file (similar to pddl file) of the image that I give as input.
An example of an HDDL is this:
(define
        (problem pfile01)
        (:domain  domain_htn)
        (:objects
                plate1 - container
                pear1 - food
                home1 wp1s wp2s - location
                wp1f - location
                robot1 - robot
        )       (:htn
                :parameters ()
                :subtasks (and
                 (task0 (move_object plate1 wp1f))
                 (task1 (move_to_container pear1 plate1))
                )
                :ordering (and
                )
        )

        (:init
                (at plate1 wp1s)
                (at pear1 wp2s)
                (at robot1 home1)
        )
)
Another example:
(define
    (problem pfile01)
    (:domain  domain_htn)
    (:objects
        tennis_ball1 - item
        white_cup1 red_cup1 - container
        banana1 pear1 - food
        home1 wp1s wp2s wp3s wp4s wp5s out_location wp1f wp2f - location
        robot1 - robot
    )
    (:htn
        :parameters ()
        :subtasks (and
            (task0 (move_object tennis_ball1 out_location))
            (task1 (move_object white_cup1 wp1f))
            (task2 (move_object red_cup1 wp2f))
            (task3 (move_to_container banana1 white_cup1))
            (task4 (move_to_container pear1 red_cup1))
        )
        :ordering (and
        )
    )

    (:init
        (at tennis_ball1 wp1s)
        (at white_cup1 wp2s)
        (at red_cup1 wp3s)
        (at banana1 wp4s)
        (at pear1 wp5s)
        (at robot1 home1)
    )
)
First, identify objects in the image and their types, including food (for example, apple, banana, etc.), containers (for example, plate, bowl, cup, basket), and other objects (listed as items).
For the location of the objects, use simply wp1s, wp2s etc, (for the start) and wp1f, wp2f etc, (for the goal).
For the goal, only food and containers are allowed on the table.
Put food in containers and remove the other object from the tables.
The task you can use are: move_object (to move the objects) and move_to_container (to move objects to the container).
To move the objects, use (move_object plate wp1f).
To remove the object, use the task (move_object, remote_control, out_location).
Only output the generated hddl languages.
"""

prompt= """
I want you to create HDDL problem file of the image that I give as input.
First, identify objects in the image and their types, including food (for example, apple, banana, etc.), containers (for example, plate, bowl, cup, basket), and other objects (listed as items).
For the location of the objects, use simply wp1s, wp2s etc, (for the start) and wp1f, wp2f etc, (for the goal); for example, (at plate1 wp1s) for the initial location of the plate1.
For the goal, only food and containers are allowed on the table.
Put food in containers and remove the other object from the tables, if they are not containers, place the food on waypoints.
The task you can use are: move_object (to move the objects) and move_to_container (to move objects to the container).
To move the objects, use (move_object object_to_move final_waypoint); to move the food, use (move_to_container food container).
Only output the generated hddl file.
"""

# Convert dataset to OAI messages
def format_data(sample):
    return {"messages": [
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
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in samples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in samples] #0 is the image; 1 is the video
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    im_start_indices = [(_label == IM_START_ID).nonzero(as_tuple=True)[0] for _label in labels]

    # the model outputs start from (im_start_indices[2]+2), which means (<im_start>assistant\n)
    start_indices = [start_idx[2] + 2 for start_idx in im_start_indices]
    # for each batch, the (:start_indices) for each batch will be set as IGNORE_INDEX
    for i, start_idx in enumerate(start_indices):
        labels[i, :start_idx+1] = IGNORE_INDEX # all input will be set as IGNORE_INDEX during fine-tuning
    # all padding tokens will be set as IGNORE_INDEX
    labels[labels == processor.tokenizer.pad_token_id] = IGNORE_INDEX

    """
    system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
    system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']

    return_message = f"{DEFAULT_IM_START_TOKEN}assistant\n"
    return_message_input_ids = processor.tokenizer(return_message, add_special_tokens=False, return_tensors='pt')['input_ids']
    """

    batch["labels"] = labels
    return batch

def generate_description(sample, model, processor):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "image","image": sample['image'], "resized_height": 320, "resized_width": 640,},
            {"type": "text", "text": prompt}
        ]},
    ]
    # Preparation for inference
    # text = processor.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )
    text = processor.apply_chat_template(
        messages, tokenize=False, return_tensors="pt"
    )
    image_inputs, video_inputs = process_vision_info(messages)
    # save the image
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    # check whether there is inf or nan in the inputs
    if torch.isnan(inputs.input_ids).any() or torch.isinf(inputs.input_ids).any():
        raise ValueError("Input contains NaN or Inf values.")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

if __name__ == "__main__":
    dataset_id = "shuooru/image-hddl-dataset"

    # dataset = [format_data(sample) for sample in data_loader.dataset]



IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]

IM_START_ID = 151644
IM_END_ID = 151645


PROMPT_0="""
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

PROMPT_1= """
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
I want you to create HDDL problem file of the image that I give as input.
First, identify objects in the image and their types, including food (for example, apple, banana, etc.), containers (for example, plate, bowl, cup, basket), and other objects (listed as items).
For the location of the objects, use simply wp1s, wp2s etc, (for the start) and wp1f, wp2f etc, (for the goal); for example, (at plate1 wp1s) for the initial location of the plate1.
For the goal, only food and containers are allowed on the table.
Put food in containers and remove the other object from the tables, if they are not containers, place the food on waypoints.
The task you can use are: move_object (to move the objects) and move_to_container (to move objects to the container).
To move the objects, use (move_object object_to_move final_waypoint); to move the food, use (move_to_container food container).
Only output the generated hddl file.
"""

PROMPT_2= """
I want you to create HDDL problem file of the image that I give as input.
First, identify objects in the image and their types, including food (for example, apple, banana, etc.), containers (for example, plate, bowl, cup, basket), and other objects (listed as items).
For the location of the objects, use simply wp1s, wp2s etc, (for the start) and wp1f, wp2f etc, (for the goal); for example, (at plate1 wp1s) for the initial location of the plate1.
For the goal, only food and containers are allowed on the table.
Put food in containers and remove the other object from the tables, if they are not containers, place the food on waypoints.
The task you can use are: move_object (to move the objects) and move_to_container (to move objects to the container).
To move the objects, use (move_object object_to_move final_waypoint); to move the food, use (move_to_container food container).
Only output the generated hddl file.
"""

PROMPT_3= """
I want you to create HDDL problem file of the image that I give as input.
For the objects:
Identify objects in the image and their types, there are classified as food (for example, apple, banana, etc.), containers (for example, plate, bowl, basket), glass (for example, cup, mug, etc.), and items (except food and containers).
Generate start and goal waypoints for all objects, for example, wp1s, wp2s etc, (for the start) and wp1f, wp2f etc, (for the goal).
For example, for 5 objects, we have 10 waypoints: wp1s, wp2s, wp3s, wp4s, wp5s (for the start) and wp1f, wp2f, wp3f, wp4f, wp5f (for the goal).
Each waypoint is of type location.
By default, there is robot1 recognized as robot.

For tasks:
Move items, containers and glass to waypoints, and move food to containers.
If they are not containers, move the food to waypoints.
The task you can use are: `move_object` (which is used to move the items, containers and glass) and `move_to_container` (to move food to the container).
The usage example is: to move items, glass and containers: (move_object object_to_move final_waypoint); to move the food: (move_to_container food container).

For initial states:
Place each object at its initial waypoint, for example, (at plate1 wp1s) for the initial location of the plate1.
If food / items / glass are in a container, use (on food/item/glass container) to indicate that the food / item / glass is in the container.
The robot1 is always at home1, use (at robot1 home1) to indicate that the robot is at home.

Only output the generated hddl file.
"""
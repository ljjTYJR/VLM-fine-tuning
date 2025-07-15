from datasets import load_dataset, Dataset, Features, Value, Image
import pandas as pd

# Load metadata
df = pd.read_csv("/media/shuo/T7/hddl_data/metadata.csv")
df = df[["image", "hddl"]]

# Load the HDDL files as strings
def load_hddl(path):
    with open(path, "r") as f:
        return f.read()
df["hddl"] = df["hddl"].apply(lambda x: load_hddl(f"/media/shuo/T7/hddl_data/{x}"))
df["image"] = df["image"].apply(lambda x: f"/media/shuo/T7/hddl_data/{x}")

# Convert to Hugging Face dataset
features = Features({
    "image": Image(),  # automatically loads as PIL image
    "hddl": Value("string")
})

dataset = Dataset.from_pandas(df, features=features)

dataset.push_to_hub("shuooru/image-hddl-dataset")
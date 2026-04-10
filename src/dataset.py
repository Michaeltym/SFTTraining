from torch.utils.data import Dataset
from src.data import EncodedData


class CustomDataset(Dataset):
    def __init__(self, encoded: list[EncodedData]) -> None:
        super().__init__()
        self.encoded = encoded

    def __getitem__(self, index: int) -> EncodedData:
        return self.encoded[index]

    def __len__(self) -> int:
        return len(self.encoded)

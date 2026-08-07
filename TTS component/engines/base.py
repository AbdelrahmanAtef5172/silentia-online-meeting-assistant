from abc import ABC, abstractmethod


class BaseProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float,
        output_path: str,
    ) -> str:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def list_voices(self) -> dict:
        ...

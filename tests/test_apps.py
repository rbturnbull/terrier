from torchapp.testing import TorchAppTestCase
from terrier.apps import Terrier


class TestTerrier(TorchAppTestCase):
    app_class = Terrier

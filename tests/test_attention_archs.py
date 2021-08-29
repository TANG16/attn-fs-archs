import unittest
import torch

from src.ndsl.architecture.attention import ConcatenateAggregator, SumAggregator, CategoricalOneHotEncoder, NumericalEncoder, TabularTransformer


class TestAggregator(unittest.TestCase):

    def test_concatenate_aggregator(self):
        agg = ConcatenateAggregator(50)
        # Parameter in format (B, S, E)
        result = agg(torch.rand((2, 10, 5)))
        self.assertEqual(
            result.size(), 
            torch.Size([2, 50]), 
            "Output shape should be (2, 50)"
        )

    def test_sum_aggregator(self):

        agg = SumAggregator(5)
        # Parameter in format (B, S, E)
        result = agg(torch.rand((2, 10, 5)))

        self.assertEqual(
            result.size(), 
            torch.Size([2, 5]), 
            "Output shape should be (2, 5)"
        )


class TestEncoder(unittest.TestCase):

    def test_oh_encoder(self):
        enc = CategoricalOneHotEncoder(10, 5)
        # Parameter in format (B, S, E)
        result = enc(torch.randint(0, 5, (2, 1)))
        self.assertEqual(
            result.size(), 
            torch.Size([2, 10]), 
            "Output shape should be (2, 10)"
        )

    def test_numerical_encoder(self):
        enc = NumericalEncoder(10)
        # Parameter in format (B, S, E)
        result = enc(torch.rand((2, 1)))

        self.assertEqual(
            result.size(), 
            torch.Size([2, 10]), 
            "Output shape should be (2, 10)"
        )

class TestTransformer(unittest.TestCase):

    def test_transformer(self):
        trans = TabularTransformer(
            n_head=2, # Number of heads per layer
            n_hid=128, # Size of the MLP inside each transformer encoder layer
            n_layers=2, # Number of transformer encoder layers    
            n_output=5, # The number of output neurons
            encoders=torch.nn.ModuleList([
                NumericalEncoder(10),
                CategoricalOneHotEncoder(10, 4),
                NumericalEncoder(10),
            ]), # List of features encoders
            dropout=0.1, # Used dropout
            aggregator=None,
        )

        input = torch.tensor([
            [0.5, 1, 0.7],
            [0.5, 3, 0.8]
        ])

        result = trans(input)

        self.assertEqual(
            result.size(), 
            torch.Size([2, 5]), 
            "Output shape should be (2, 10)"
        )






if __name__ == '__main__':
    unittest.main()
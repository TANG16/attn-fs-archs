import torch
import torch.nn as nn

class BaseAggregator(nn.Module):
    def __init__(self, output_size):
        super(BaseAggregator, self).__init__()
        self.output_size = output_size

    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")

class ConcatenateAggregator(BaseAggregator):
    def forward(self, src):
        return torch.flatten(src, start_dim=1)

class SumAggregator(BaseAggregator):
    def forward(self, src):
        return torch.sum(src, dim=1, keepdim=False)


class FeatureEncoder(nn.Module):
    def __init__(self, output_size):
        super(FeatureEncoder, self).__init__()
        self.output_size = output_size
    
    def forward(self, src):
        raise NotImplementedError("This feature hasn't been implemented yet!")


class CategoricalOneHotEncoder(FeatureEncoder):
    def __init__(self, output_size, n_labels):
        super(CategoricalOneHotEncoder, self).__init__(output_size)
        self.output_size = output_size
        self.n_labels = n_labels
        self.embedding = nn.utils.weight_norm(nn.Linear(n_labels, output_size))

    def forward(self, src):
        inp = torch.ones((src.size()[0], self.n_labels))
        inp[:, src.size()[1]] = 1.0
        return self.embedding(inp)

class NumericalEncoder(FeatureEncoder):
    def __init__(self, output_size):
        super(NumericalEncoder, self).__init__(output_size)
        self.output_size = output_size
        self.embedding = nn.utils.weight_norm(nn.Linear(1, output_size))

    def forward(self, src):
        return self.embedding(src)
        

class TabularTransformer(nn.Module):
    
    def __init__(
        self, 
        n_head, # Number of heads per layer
        n_hid, # Size of the MLP inside each transformer encoder layer
        n_layers, # Number of transformer encoder layers    
        n_output, # The number of output neurons
        encoders, # List of features encoders
        dropout=0.1, # Used dropout
        aggregator=None, # The aggregator for output vectors before decoder
        ):


        super(TabularTransformer, self).__init__()

        # Verify that encoders are correct
        if not isinstance(encoders, nn.ModuleList):
            raise TypeError("Parameter encoders must be an instance of torch.nn.ModuleList")

        # Embedding size
        self.n_input = None

        for idx, encoder in enumerate(encoders):
            
            if not issubclass(type(encoder), FeatureEncoder):
                raise TypeError("All encoders must inherit from FeatureEncoder. Invalid index {}".format(idx))

            if self.n_input is None:
                self.n_input = encoder.output_size
            elif self.n_input != encoder.output_size:
                raise ValueError("All encoders must have the same output")

        self.encoders = encoders
        # Building transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(self.n_input, n_head, n_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # The default aggregator will be ConcatenateAggregator
        if aggregator is None:
            self.aggregator = ConcatenateAggregator(self.n_input * len(self.encoders))
        else:
            self.aggregator = aggregator

        # Validates that aggregator inherit from BaseAggregator
        if not issubclass(type(self.aggregator), BaseAggregator):
            raise TypeError("Parameter aggregator must inherit from BaseAggregator")

        self.decoder = nn.Linear(self.aggregator.output_size, n_output)

    def forward(self, src):
        # src came with two dims: (batch_size, num_features)
        embeddings = []

        # Computes embeddings for each feature
        for ft_idx, encoder in enumerate(self.encoders):
            # Each encoder must return a two dims tensor (batch, embedding_size)
            encoding = encoder(src[:, ft_idx].unsqueeze(1))
            embeddings.append(encoding)

        # embeddings has 3 dimensions (num_features, batch, embedding_size)
        embeddings = torch.stack(embeddings)
        # Encodes through transformer encoder
        # Due transpose, the output will be in format (batch, num_features, embedding_size)
        output = self.transformer_encoder(embeddings).transpose(0, 1)

        # Aggregation of encoded vectors
        output = self.aggregator(output)

        # Decoding
        output = self.decoder(output)
        
        return output


class MixtureModelv0(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv0, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return attn_mat.sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output


class MixtureModelv1(nn.Module):

    def __init__(self, ninp, nhead, nhid, nmodels, nfeatures, nclasses, dropout=0.5):
        super(MixtureModelv1, self).__init__()

        self.attention_mechanism = nn.MultiheadAttention(
                                        ninp, 
                                        nhead, 
                                        dropout=dropout
                                    )

                    
        self.nfeatures = nfeatures
        self.nmodels = nmodels
        #self.num_embedding = nn.Linear(1, ninp)

        self.embedding = nn.ModuleList()
        
        for feature in range(nfeatures):
            self.embedding.append(nn.utils.weight_norm(nn.Linear(1, ninp)))
        

        self.representation = nn.Sequential(
                                nn.Linear(nfeatures * ninp, nhid),
                                nn.BatchNorm1d(nhid),                          
                                nn.Dropout(dropout)
                            )


        self.model_weighting = nn.Sequential(
                                    nn.Linear(nfeatures, nmodels),
                                    nn.Softmax(dim=-1)
                                )
        
        self.models = nn.ModuleList()

        self.aggregator = nn.Linear(nfeatures, nfeatures)
        
        for model in range(nmodels):
            self.models.append(nn.Linear(nhid, nclasses))
        
    def aggregate(self, attn_mat):
        return self.aggregator(attn_mat).sum(dim=1)


    def forward(self, src):
        
        #src = self.num_embedding(src)
        src_nums = []
        
        for feature in range(self.nfeatures):
            src_nums.append(
                self.embedding[feature](src[:, feature]).unsqueeze(1)
            )
        
        #src_num = self.num_embedding(src[:, len(categorical_cols):])
        src = torch.cat(src_nums, dim=1)
        src = src.transpose(0, 1)

        attn_out, attn_mat = self.attention_mechanism(src, src, src)

        attn_out = attn_out.transpose(0, 1).flatten(start_dim=1)
        attn_mat = self.aggregate(attn_mat)

        representation = self.representation(attn_out)

        model_weights = self.model_weighting(attn_mat).unsqueeze(1)

        outputs = []
        
        for model in range(self.nmodels):
            outputs.append(
                self.models[model](representation)
            )

        output = torch.stack(outputs, dim=0).transpose(0, 1)
        output = torch.bmm(model_weights, output).sum(dim=1)

        return output


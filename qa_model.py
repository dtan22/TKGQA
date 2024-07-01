import math
import torch
from torch import nn
import numpy as np
from tcomplex import TComplEx
from transformers import BertModel, DistilBertModel
from torch.nn import LayerNorm
import copy
import torch.nn.functional as F
import pdb
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from torch.nn import LayerNorm
# training data: questions
# model:
# 1. tkbc model embeddings (may or may not be frozen)
# 2. question sentence embeddings (may or may not be frozen)
# 3. linear layer to project question embeddings (unfrozen)
# 4. transformer that takes these embeddings (unfrozen) (cats them along a dimension, also takes a mask)
# 5. average output embeddings of transformer or take last token embedding?
# 6. linear projection of this embedding to tkbc embedding dimension
# 7. score with all possible entities/times and sigmoid
# 8. BCE loss (multiple correct possible)

class QA_TempoQR(nn.Module):
	def __init__(self, tkbc_model, args):
		super().__init__()
		self.model = args.model
		self.supervision = args.supervision
		self.extra_entities = args.extra_entities
		self.fuse = args.fuse
		self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
		self.sentence_embedding_dim = 768  # hardwired from 
        	
		self.pretrained_weights = args.lm_model
		if args.lm_model=='distilbert-base-uncased':
			self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
		elif args.lm_model == 'bert-base-multilingual-cased':
			self.lm_model = BertModel.from_pretrained(self.pretrained_weights)
		if args.lm_frozen == 1:
				print('Freezing LM params')
				for param in self.lm_model.parameters():
					param.requires_grad = False
		else:
				print('Unfrozen LM params')

        
		# transformer
		self.transformer_dim = self.tkbc_embedding_dim  # keeping same so no need to project embeddings
		self.nhead = 8
		self.num_layers = 6
		self.transformer_dropout = 0.1
		self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead,
																		dropout=self.transformer_dropout)
		encoder_norm = LayerNorm(self.transformer_dim)
		self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
																			norm=encoder_norm)
		self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
		self.project_entity = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		
		#TKG embeddings
		self.tkbc_model = tkbc_model
		num_entities = tkbc_model.embeddings[0].weight.shape[0]
		num_times = tkbc_model.embeddings[2].weight.shape[0]
		ent_emb_matrix = tkbc_model.embeddings[0].weight.data
		time_emb_matrix = tkbc_model.embeddings[2].weight.data
        
        
		full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
		# +1 is for padding idx
		self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
																self.tkbc_embedding_dim,
																padding_idx=num_entities + num_times)
		self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)
        

		if args.frozen == 1:
				print('Freezing entity/time embeddings')
				self.entity_time_embedding.weight.requires_grad = False
				for param in self.tkbc_model.parameters():
					param.requires_grad = False
		else:
				print('Unfrozen entity/time embeddings')

		# position embedding for transformer
		self.max_seq_length = 100  # randomly defining max length of tokens for question
		self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)
		# print('Random starting embedding')
		self.loss = nn.CrossEntropyLoss(reduction='mean')
		self.layer_norm = nn.LayerNorm(self.transformer_dim)
        

		self.linear = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
		self.linearT = nn.Linear(768, self.tkbc_embedding_dim)  # to project question embedding
		self.lin_cat = nn.Linear(3*self.transformer_dim, self.transformer_dim)

		self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
		self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        
		self.dropout = torch.nn.Dropout(0.3)
		self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
		self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        
		return


         
    
	def invert_binary_tensor(self, tensor):
		ones_tensor = torch.ones(tensor.shape, dtype=torch.float32).cuda()
		inverted = ones_tensor - tensor
		return inverted


	def infer_time(self, head_embedding, tail_embedding, relation_embedding):
		lhs = head_embedding
		rhs = tail_embedding
		rel = relation_embedding

		time = self.tkbc_model.embeddings[2].weight  # + self.tkbc_model.lin2(self.tkbc_model.time_embedding.weight)
		# time = self.entity_time_embedding.weight

		lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		return torch.cat([
					(lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
					lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]),
					(lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
					lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1])], dim=-1
		)
        
	# scoring function from TComplEx
	def score_time(self, head_embedding, tail_embedding, relation_embedding):
		lhs = head_embedding
		rhs = tail_embedding
		rel = relation_embedding

		time = self.tkbc_model.embeddings[2].weight 

		lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		return (
					(lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
					lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
					(lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
					lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
		)

	def score_entity(self, head_embedding, tail_embedding, relation_embedding, time_embedding):

		lhs = head_embedding[:, :self.tkbc_model.rank], head_embedding[:, self.tkbc_model.rank:]
		rel = relation_embedding
		time = time_embedding

		rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
		time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

		right = self.tkbc_model.embeddings[0].weight 
		# right = self.entity_time_embedding.weight
		right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

		rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
		full_rel = rt[0] - rt[3], rt[1] + rt[2]

		return (
					(lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
					(lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
		)
    

	def forward(self, a):
		#Tokenized questions, where entities are masked from the sentence to have TKG embeddings
		question_tokenized = a[0].cuda()
		question_attention_mask = a[1].cuda()
		entities_times_padded = a[2].cuda()
		entity_mask_padded = a[3].cuda()
        
		#Annotated entities/timestamps
		heads = a[4].cuda()
		tails = a[5].cuda()
		times = a[6].cuda()
        
		#t1 and t2 by Hard Supervision
		t1 = a[7].cuda()
		t2 = a[8].cuda()
        
		#One extra entity for new before & after question type
		tails2 = a[9].cuda()
        
		#TKG embeddings
		head_embedding = self.entity_time_embedding(heads)
		tail_embedding = self.entity_time_embedding(tails)
		tail_embedding2 = self.entity_time_embedding(tails2)
		time_embedding = self.entity_time_embedding(times)
        
		#Hard Supervision
		t1_emb = self.tkbc_model.embeddings[2](t1) 
		t2_emb = self.tkbc_model.embeddings[2](t2) 

        
        
		#entity embeddings to replace in sentence
		entity_time_embedding = self.entity_time_embedding(entities_times_padded)
        
		#context-aware step
		outputs = self.lm_model(question_tokenized, attention_mask=question_attention_mask)
		last_hidden_states = outputs.last_hidden_state
        
		if self.supervision == 'soft':
		#infer time embeddings for t1 and t2 with Soft Supervision
			lm_last_hidden_states = outputs[0]
			states = lm_last_hidden_states.transpose(1,0)
			cls_embedding = states[0]
			cls_embedding = self.linearT(cls_embedding)
		    
			t1_emb = self.infer_time(head_embedding, tail_embedding, cls_embedding)
			t2_emb = self.infer_time(tail_embedding, head_embedding, cls_embedding)

			if self.extra_entities == True and not self.training:   #this is for the created before & after questions - we expect one more entity
				t3_emb = self.infer_time(tail_embedding2, head_embedding, cls_embedding)
				t2_emb = t2_emb+t3_emb #for simplicity, can be omitted and treated as t3_emb

		#entity-aware step
		question_embedding = self.project_sentence_to_transformer_dim(last_hidden_states)
		entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding.shape)
		masked_question_embedding = question_embedding * entity_mask  # set entity positions 0

		entity_time_embedding_projected = self.project_entity(entity_time_embedding)
        
        
        
		#time-aware step 
		if self.model == 'tempoqr':
			time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0,1)
			time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)
		    
			time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0,1)
			time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)
			if self.fuse == 'cat':
				entity_time_embedding_projected = self.lin_cat(torch.cat((entity_time_embedding_projected, time_pos_embeddings1,time_pos_embeddings2), dim=-1))
			else:
				entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2
        
		# Transformer information fusion layer
		masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask)
        
        
		combined_embed = masked_question_embedding + masked_entity_time_embedding
		# also need to add position embedding
		sequence_length = combined_embed.shape[1]
		v = np.arange(0, sequence_length, dtype=np.long)
		indices_for_position_embedding = torch.from_numpy(v).cuda()
		position_embedding = self.position_embedding(indices_for_position_embedding)
		position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)
        

		combined_embed = combined_embed + position_embedding 

		combined_embed = self.layer_norm(combined_embed)
		combined_embed = torch.transpose(combined_embed, 0, 1)

		mask2 = ~(question_attention_mask.bool()).cuda()

		output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)
        
		# Answer Predictions
		if self.model == 'cronkgqa': #no transformer, go back and get cls_embedding
			relation_embedding = cls_embedding
		else:
			relation_embedding = output[0]# self.linear(output[0]) #cls token embedding
		relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
		relation_embedding2 = self.dropout(self.bn1(self.linear2(relation_embedding)))


        
		scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)

		if self.model == 'cronkgqa' or (self.model == 'entityqr' and self.supervision != 'none'): #supervision for cronkgqa and entityqr
			time_embedding = (time_embedding + t1_emb + t2_emb)/3   #just take the mean
		scores_entity1 = self.score_entity(head_embedding, tail_embedding, relation_embedding2, time_embedding)
		scores_entity2 = self.score_entity(tail_embedding, head_embedding, relation_embedding2, time_embedding)
		scores_entity = torch.maximum(scores_entity1, scores_entity2)
        
		scores = torch.cat((scores_entity, scores_time), dim=1)
        
		return scores  

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def select(matrix, top_num):
    batch = matrix.size(0)
    len = matrix.size(1)
    matrix = matrix.reshape(batch, -1)
    maxk, _ = torch.topk(matrix, top_num, dim=1)

    for i in range(batch):
        matrix[i] = (matrix[i] >= maxk[i][-1])
    matrix = matrix.reshape(batch, len, len)
    matrix = matrix + matrix.transpose(-2, -1)

    # selfloop
    for i in range(batch):
        matrix[i].fill_diagonal_(1)

    return matrix

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % head_num == 0

        self.d_k = int(hidden_dim // head_num)
        self.head_num = head_num
        self.linears = clones(nn.Linear(hidden_dim, hidden_dim), 2)
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)

        b = ~score_mask[:, :, :, 0:1]
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)

        return attn

class GCN(nn.Module):
    def __init__(self, args, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = 768
        self.linearc = nn.Linear(766, 768)

        self.fc = nn.Sequential(
            nn.Linear(768, 768, bias=False),
            nn.ReLU(),
            nn.Linear(768, 768, bias=False)
        )

        self.in_drop = nn.Dropout(args.input_dropout)
        self.gcn_drop = nn.Dropout(args.gcn_dropout)

        self.W = nn.ModuleList()
        self.attn = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim + layer * self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
            self.attn.append(MultiHeadAttention(3, input_dim)) if layer != 0 else None

    def GCN_layer(self, adj, gcn_inputs, denom, l):
        Ax = adj.bmm(gcn_inputs)
        AxW = self.W[l](Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW) + self.W[l](gcn_inputs)
        gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        return gcn_inputs

    def forward(self, adj, inputs, score_mask, type):
        denom = adj.sum(2).unsqueeze(2) + 1
        n = inputs.size(1)

        Convolution = nn.Conv1d(n, n, 3).cuda()
        out = self.GCN_layer(adj, inputs, denom, 0)

        conve = F.relu(Convolution(out))
        out = self.linearc(conve)

        for i in range(1, self.layers):
            inputs = torch.cat((inputs, out), dim=-1)
            if type == 'semantic':
                adj = self.attn[i - 1](inputs, inputs, score_mask)
                probability = F.softmax(adj.sum(dim=(-2, -1)), dim=0)
                max_idx = torch.argmax(probability, dim=1)
                adj = torch.stack([adj[i][max_idx[i]] for i in range(len(max_idx))], dim=0)
                adj = select(adj, 2) * adj
                denom = adj.sum(2).unsqueeze(2) + 1
            out = self.GCN_layer(adj, inputs, denom, i)
            out = self.fc(out)
        return out

class CTRN(nn.Module):
    def __init__(self, tkbc_model, args):
        super().__init__()
        self.model = args.model
        self.supervision = args.supervision
        self.extra_entities = args.extra_entities
        self.fuse = args.fuse
        self.tkbc_embedding_dim = tkbc_model.embeddings[0].weight.shape[1]
        self.sentence_embedding_dim = 768  # hardwired from

        self.pretrained_weights = args.lm_model
	if args.lm_model=='distilbert-base-uncased':
        	self.lm_model = DistilBertModel.from_pretrained(self.pretrained_weights)
	elif args.lm_model=='bert-base-multilingual-cased':
		self.lm_model = BertModel.from_pretrained(self.pretrained_weights)
        if args.lm_frozen == 1:
            print('Freezing LM params')
            for param in self.lm_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen LM params')

        self.transformer_dim = self.tkbc_embedding_dim
        self.nhead = 8
        self.num_layers = 6
        self.transformer_dropout = 0.1
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.nhead,
                                                        dropout=self.transformer_dropout)
        encoder_norm = LayerNorm(self.transformer_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers,
                                                         norm=encoder_norm)
        self.project_sentence_to_transformer_dim = nn.Linear(self.sentence_embedding_dim, self.transformer_dim)
        self.project_entity = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)

        self.tkbc_model = tkbc_model
        num_entities = tkbc_model.embeddings[0].weight.shape[0]
        num_times = tkbc_model.embeddings[2].weight.shape[0]
        ent_emb_matrix = tkbc_model.embeddings[0].weight.data
        time_emb_matrix = tkbc_model.embeddings[2].weight.data

        full_embed_matrix = torch.cat([ent_emb_matrix, time_emb_matrix], dim=0)
        self.entity_time_embedding = nn.Embedding(num_entities + num_times + 1,
                                                  self.tkbc_embedding_dim,
                                                  padding_idx=num_entities + num_times)
        self.entity_time_embedding.weight.data[:-1, :].copy_(full_embed_matrix)

        if args.frozen == 1:
            print('Freezing entity/time embeddings')
            self.entity_time_embedding.weight.requires_grad = False
            for param in self.tkbc_model.parameters():
                param.requires_grad = False
        else:
            print('Unfrozen entity/time embeddings')

        self.max_seq_length = 100
        self.position_embedding = nn.Embedding(self.max_seq_length, self.tkbc_embedding_dim)

        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.layer_norm = nn.LayerNorm(self.transformer_dim)

        self.linear = nn.Linear(self.sentence_embedding_dim, self.tkbc_embedding_dim)
        self.linearT = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.lin_cat = nn.Linear(3 * self.transformer_dim, self.transformer_dim)

        self.linear1 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.linear2 = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.lineart = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.linearr = nn.Linear(self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.tkbc_embedding_dim)

        self.combine_all_entities_func_forReal = nn.Linear(self.tkbc_embedding_dim, self.tkbc_model.rank)
        self.combine_all_entities_func_forCmplx = nn.Linear(self.tkbc_embedding_dim, self.tkbc_model.rank)
        self.combine_all_times_func_forReal = nn.Linear(self.tkbc_embedding_dim, self.tkbc_model.rank)
        self.combine_all_times_func_forCmplx = nn.Linear(self.tkbc_embedding_dim, self.tkbc_model.rank)
        self.combine_relation = nn.Linear(2*self.tkbc_embedding_dim, self.tkbc_embedding_dim)

        self.Convolution = nn.Conv1d(1, 1, 3)
        self.linearc = nn.Linear(766, 512)

        self.Line = nn.Linear(self.sentence_embedding_dim, self.tkbc_embedding_dim)
        self.kg_gate = nn.Linear(2 * self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.kg_gate1 = nn.Linear(2 * self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.COM = nn.Linear(2 * self.tkbc_embedding_dim, self.tkbc_embedding_dim)
        self.attn = MultiHeadAttention(3, self.sentence_embedding_dim)
        self.gcn_common = GCN(args, self.sentence_embedding_dim, 2)

    def invert_binary_tensor(self, tensor):
        ones_tensor = torch.ones(tensor.shape, dtype=torch.float32).cuda()
        inverted = ones_tensor - tensor
        return inverted

    def infer_time(self, head_embedding, tail_embedding, relation_embedding):
        lhs = head_embedding
        rhs = tail_embedding
        rel = relation_embedding

        time = self.tkbc_model.embeddings[2].weight

        lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        return torch.cat([
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]),
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1])], dim=-1
        )

    def getQuestionEmbedding(self, question_tokenized, attention_mask):
        roberta_states = self.lm_model(question_tokenized, attention_mask=attention_mask)
        question_embedding = roberta_states[0]
        states = question_embedding.transpose(1, 0)
        cls_embedding = states[0]
        return question_embedding, cls_embedding

    def score_time(self, head_embedding, tail_embedding, relation_embedding):
        lhs = head_embedding
        rhs = tail_embedding
        rel = relation_embedding

        time = self.tkbc_model.embeddings[2].weight

        lhs = lhs[:, :self.tkbc_model.rank], lhs[:, self.tkbc_model.rank:]
        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        return (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def score_time1(self, head_embedding, tail_embedding, relation_embedding):
        lhs = self.combine_all_times_func_forReal(torch.cat((head_embedding[:, :self.tkbc_model.rank],
                                                             tail_embedding[:, :self.tkbc_model.rank]), dim=1)) \
            , self.combine_all_times_func_forCmplx(torch.cat((head_embedding[:, self.tkbc_model.rank:],
                                                              tail_embedding[:, self.tkbc_model.rank:]), dim=1))

        rel = relation_embedding

        time = self.tkbc_model.embeddings[2].weight

        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        rhs = rhs[:, :self.tkbc_model.rank], rhs[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        return (
            (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
             lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
            (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
             lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
        )

    def score_entity(self, head_embedding, tail_embedding, relation_embedding, time_embedding):
        lhs = head_embedding[:, :self.tkbc_model.rank], head_embedding[:, self.tkbc_model.rank:]
        rel = relation_embedding
        time = time_embedding

        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]
        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        right = self.tkbc_model.embeddings[0].weight
        right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

    def score_entity1(self, head_embedding, tail_embedding, relation_embedding, time_embedding):
        lhs = self.combine_all_entities_func_forReal(torch.cat((head_embedding[:, :self.tkbc_model.rank],
                                                                tail_embedding[:, :self.tkbc_model.rank]),
                                                               dim=1)) \
            , self.combine_all_entities_func_forCmplx(torch.cat((head_embedding[:, self.tkbc_model.rank:],
                                                                 tail_embedding[:, self.tkbc_model.rank:]),
                                                                dim=1))

        rel = relation_embedding

        time = time_embedding

        rel = rel[:, :self.tkbc_model.rank], rel[:, self.tkbc_model.rank:]

        time = time[:, :self.tkbc_model.rank], time[:, self.tkbc_model.rank:]

        right = self.tkbc_model.embeddings[0].weight
        right = right[:, :self.tkbc_model.rank], right[:, self.tkbc_model.rank:]

        rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
        full_rel = rt[0] - rt[3], rt[1] + rt[2]

        return (
            (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
            (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
        )

    def inputs_to_att_adj(self, input, score_mask):
        attn_tensor = self.attn(input, input, score_mask)
        attn_tensor = torch.sum(attn_tensor, dim=1)
        return attn_tensor

    def forward(self, a):
        question_tokenized = a[0].cuda()
        question_attention_mask = a[1].cuda()
        entities_times_padded = a[2].cuda()
        entity_mask_padded = a[3].cuda()
        heads = a[4].cuda()
        tails = a[5].cuda()
        times = a[6].cuda()

        head_embedding = self.entity_time_embedding(heads)
        tail_embedding = self.entity_time_embedding(tails)
        time_embedding = self.entity_time_embedding(times)

        entity_time_embedding = self.entity_time_embedding(entities_times_padded)

        question_embedding, cls_embedding = self.getQuestionEmbedding(question_tokenized, question_attention_mask)
        score_mask = torch.matmul(question_embedding, question_embedding.transpose(-2, -1))
        score_mask = (score_mask == 0)
        score_mask = score_mask.unsqueeze(1).repeat(1, 3, 1, 1).cuda()

        att_adj = self.inputs_to_att_adj(question_embedding, score_mask)
        h_cse = self.gcn_common(att_adj, question_embedding, score_mask, 'semantic')
        n = question_embedding.size(1)
        Convolution = nn.Conv1d(n, n, 3).cuda()
        conve = F.relu(Convolution(h_cse))
        deep_q = self.linearc(conve)
        asp_wn = question_attention_mask.sum(dim=1).unsqueeze(-1)
        mask = question_attention_mask.unsqueeze(-1).repeat(1, 1, 512)
        h_e = (deep_q * mask).sum(dim=1) / asp_wn
        question_embedding1 = self.project_sentence_to_transformer_dim(question_embedding)
        entity_mask = entity_mask_padded.unsqueeze(-1).expand(question_embedding1.shape)
        entity_time_embedding_projected = self.project_entity(entity_time_embedding)

        if self.supervision == 'soft':
            cls = self.linear(cls_embedding)
            gate_value = self.kg_gate1(torch.cat([h_e, cls], dim=-1)).sigmoid()
            vq = gate_value * h_e + (1 - gate_value) * cls
            cls_embedding = self.linearT(vq)

            t1_emb = self.infer_time(head_embedding, tail_embedding, cls_embedding)
            t2_emb = self.infer_time(tail_embedding, head_embedding, cls_embedding)
            time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0, 1)
            time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)

            time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0, 1)
            time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)
            if self.fuse == 'cat':
                entity_time_embedding_projected = self.lin_cat(
                    torch.cat((entity_time_embedding_projected, time_pos_embeddings1, time_pos_embeddings2), dim=-1))
            else:
                entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2
        elif self.supervision == 'hard':
            t1 = a[7].cuda()
            t2 = a[8].cuda()
            t1_emb = self.tkbc_model.embeddings[2](t1)
            t2_emb = self.tkbc_model.embeddings[2](t2)
            time_pos_embeddings1 = t1_emb.unsqueeze(0).transpose(0, 1)
            time_pos_embeddings1 = time_pos_embeddings1.expand(entity_time_embedding_projected.shape)

            time_pos_embeddings2 = t2_emb.unsqueeze(0).transpose(0, 1)
            time_pos_embeddings2 = time_pos_embeddings2.expand(entity_time_embedding_projected.shape)
            if self.fuse == 'cat':
                entity_time_embedding_projected = self.lin_cat(
                    torch.cat((entity_time_embedding_projected, time_pos_embeddings1, time_pos_embeddings2), dim=-1))
            else:
                entity_time_embedding_projected = entity_time_embedding_projected + time_pos_embeddings1 + time_pos_embeddings2

        masked_entity_time_embedding = entity_time_embedding_projected * self.invert_binary_tensor(entity_mask)
        combined_embed = question_embedding1 + masked_entity_time_embedding

        sequence_length = combined_embed.shape[1]
        v = np.arange(0, sequence_length, dtype=np.long)
        indices_for_position_embedding = torch.from_numpy(v).cuda()
        position_embedding = self.position_embedding(indices_for_position_embedding)
        position_embedding = position_embedding.unsqueeze(0).expand(combined_embed.shape)

        combined_embed = combined_embed + position_embedding
        combined_embed = self.layer_norm(combined_embed)
        combined_embed = torch.transpose(combined_embed, 0, 1)

        mask2 = ~(question_attention_mask.bool()).cuda()
        output = self.transformer_encoder(combined_embed, src_key_padding_mask=mask2)

        embedding = output.transpose(1, 0)
        gate_value = self.kg_gate(torch.cat([embedding, deep_q], dim=-1)).sigmoid()
        fun_embedding = gate_value * embedding + (1 - gate_value) * deep_q
        relation_embedding = fun_embedding.transpose(1, 0)[0]

        relation_embedding1 = self.dropout(self.bn1(self.linear1(relation_embedding)))
        relation_embedding2 = self.dropout(self.bn1(self.linear2(relation_embedding)))

        scores_time = self.score_time(head_embedding, tail_embedding, relation_embedding1)
        scores_entity1 = self.score_entity(head_embedding, tail_embedding, relation_embedding2, time_embedding)
        scores_entity2 = self.score_entity(tail_embedding, head_embedding, relation_embedding2, time_embedding)
        scores_entity3 = self.score_entity1(head_embedding, tail_embedding, relation_embedding2, time_embedding)

        scores_entity4 = torch.maximum(scores_entity1, scores_entity2)
        scores_entity = torch.maximum(scores_entity3, scores_entity4)

        scores = torch.cat((scores_entity, scores_time), dim=1)

        return scores

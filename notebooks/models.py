import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze first 8 layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(cls_output)
        return logits

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Freeze first 8 layers
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
                
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(cls_output)
        return logits

class HybridBERT_RoBERTa(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(HybridBERT_RoBERTa, self).__init__()
        
        # 1. Initialize Body 1 (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 2. Initialize Body 2 (RoBERTa)
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # 3. Freeze the first 8 layers of both models (to prevent overfitting and speed up training)
        # BERT has 12 layers in `encoder.layer`
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
                
        # RoBERTa has 12 layers in `encoder.layer`
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in self.roberta.encoder.layer[i].parameters():
                param.requires_grad = False
                
        # 4. Projection Layers (Dimensionality Reduction 768 -> 256)
        self.bert_projection = nn.Linear(768, 256)
        self.roberta_projection = nn.Linear(768, 256)
        
        # 5. Regularization (Dropout + BatchNorm1d)
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm = nn.BatchNorm1d(512) # 256 (BERT) + 256 (RoBERTa) = 512
        
        # 6. Classification Head
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, input_ids_bert, attention_mask_bert, input_ids_roberta, attention_mask_roberta):
        # Forward pass through BERT
        bert_outputs = self.bert(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
        bert_cls = bert_outputs.pooler_output # [CLS] token representation (Batch, 768)
        
        # Forward pass through RoBERTa
        roberta_outputs = self.roberta(input_ids=input_ids_roberta, attention_mask=attention_mask_roberta)
        roberta_cls = roberta_outputs.pooler_output # [CLS] token representation (Batch, 768)
        
        # Project down to 256
        bert_proj = self.dropout(self.bert_projection(bert_cls))
        roberta_proj = self.dropout(self.roberta_projection(roberta_cls))
        
        # Feature-level Fusion (Concatenation)
        fused_features = torch.cat((bert_proj, roberta_proj), dim=1) # (Batch, 512)
        
        # Ensure the shape is exactly what we expect (batch_size, 512)
        assert fused_features.shape[1] == 512, f"Expected 512 features, got {fused_features.shape[1]}"
        
        # Apply Batch Normalization
        fused_features = self.batch_norm(fused_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits

# Sanity Check
if __name__ == '__main__':
    print("Testing Hybrid Model initialization...")
    model = HybridBERT_RoBERTa(num_classes=11)
    
    # Fake inputs (Batch size = 2, Seq Len = 16)
    fake_input_ids_b = torch.randint(0, 1000, (2, 16))
    fake_mask_b = torch.ones((2, 16))
    
    fake_input_ids_r = torch.randint(0, 1000, (2, 16))
    fake_mask_r = torch.ones((2, 16))
    
    print("Running forward pass...")
    logits = model(fake_input_ids_b, fake_mask_b, fake_input_ids_r, fake_mask_r)
    
    print(f"Output logits shape: {logits.shape}") # Expected (2, 11)
    print("Hybrid Model Sanity Check Passed!")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import random
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim, attention_type='general'):
        super().__init__()
        self.attention_type = attention_type
        self.linear = nn.Linear(enc_hidden_dim, dec_hidden_dim, bias=False)


    def forward(self, decoder_hidden, encoder_outputs, src_mask=None):
        if self.attention_type.lower() == 'general':
            projected = self.linear(encoder_outputs)

            decoder_hidden_unsqueezed = decoder_hidden.unsqueeze(2)
            scores = torch.bmm(projected, decoder_hidden_unsqueezed).squeeze(2)

            if src_mask is not None:
                scores = scores.masked_fill(src_mask == 0, float('-inf'))

            attention_weights = F.softmax(scores, dim=1)
            attention_weights_unsqueezed = attention_weights.unsqueeze(1)

            context = torch.bmm(attention_weights_unsqueezed, encoder_outputs).squeeze(1)

            return context, attention_weights

        elif self.attention_type.lower() == 'concat':
            pass
        else:
            raise ValueError(f'Передан некорректный аргумент attention_type: {self.attention_type}.')


class AttentionSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5, src_mask=None):
        encoder_outputs, hidden, cell = self.encoder(src)

        tgt_len = tgt.shape[1]
        batch_size = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for i in range(tgt_len):
            logits, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs, src_mask
            )

            outputs[:, i, :] = logits
            if i < tgt_len - 1:
                if random.random() < teacher_forcing_ratio:
                    decoder_input = tgt[:, i + 1].unsqueeze(1)
                else:
                    decoder_input = logits.argmax(dim=-1).unsqueeze(1)

        return outputs


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size=20000,
        embedding_dim=128,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=tgt_vocab_size)
        self.attention = Attention(enc_hidden_dim=hidden_dim, dec_hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combined_fc = nn.Linear(hidden_dim + hidden_dim, hidden_dim)

    def forward(self, x, hidden, cell, encoder_outputs, src_mask=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        # 1. Сначала шаг LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        # 2. Рассчитываем внимание от нового скрытого состояния
        current_hidden = hidden[-1]  # [batch, hidden_dim]
        context, attention_weights = self.attention(
            current_hidden, encoder_outputs, src_mask
        )

        # 3. Объединяем контекст и выход LSTM
        combined = torch.cat([lstm_output.squeeze(1), context], dim=-1)
        combined = torch.tanh(self.combined_fc(combined))

        # 4. Получаем логиты
        logits = self.fc_out(combined)
        return logits, hidden, cell, attention_weights


class AttentionEncoder(nn.Module):
    def __init__(
        self,
        src_vocab_size=20000,
        embedding_dim=128,
        hidden_dim=512,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return output, hidden, cell  # output будет нужен для attention


class Vocabulary():
    def __init__(self,
                 min_freq: int = 1,
                 max_size: int = 10000):
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0 : '<pad>', 1 : '<unk>', 2 : '<sos>', 3 : '<eos>'}
        self.max_size = max_size
        self.min_freq = min_freq
        self.word_freq = defaultdict(int)


    def tokens_from_file(self, filename, encoding='utf-8') -> list:
        all_tokens = []

        with open(filename, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = line.split()

                all_tokens.extend(tokens)
        return all_tokens


    def add_word(self, word: str) -> int:
        """Добавляет новое слово в словари преобразования.

        Args:
            word: Слово для добавления.

        Returns:
            int: Индекс добавленного или уже существующего слова.
        """
        # Если слово не в словаре - добавляем
        if word not in self.word2idx:
            idx = len(self.idx2word) # индекс нового слова
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        return self.word2idx[word]


    def build_vocab(self, file_path = None, encoding='utf-8') -> 'Vocabulary':
        all_tokens = self.tokens_from_file(file_path, encoding)

        # Инициализация словаря частот
        for token in all_tokens:
            self.word_freq[token] += 1

        # Фильтрация по min_freq
        existing_words = set(self.word2idx.keys())
        filtered_words = [
            word for word, freq in self.word_freq.items()
            if freq >= self.min_freq and word not in existing_words
        ]

        # Сортировка по убыванию
        filtered_words.sort(key=lambda x: self.word_freq[x], reverse=True)

        # проверка размера словаря
        remaining = self.max_size - len(self.word2idx)
        if remaining > 0 and (len(filtered_words) > remaining):
            filtered_words = filtered_words[:remaining]
        elif remaining <= 0:
            filtered_words = []
            print('Предупреждение: Vocabulary достиг максимального размера.')
            print('Новые слова не были добавлены')

        # Добавление в словарь
        for word in filtered_words:
            self.add_word(word)

        return self


    def encode_file(self, file_path, encoding='utf-8', add_special_tokens: bool = True) -> list:
        tokens = self.tokens_from_file(file_path, encoding)
        indexed_text = [self.word2idx.get(token, self.word2idx['<unk>']) \
                        for token in tokens]
        if add_special_tokens:
            indexed_text.append(self.word2idx['<eos>'])
            indexed_text.insert(0, self.word2idx['<sos>'])

        return indexed_text


    def encode_tokens(self, tokens, add_special_tokens=True):
        indexed_text = [self.word2idx.get(token, self.word2idx['<unk>']) \
                        for token in tokens]
        if add_special_tokens:
            indexed_text.append(self.word2idx['<eos>'])
            indexed_text.insert(0, self.word2idx['<sos>'])
        return indexed_text


    def decode(self,
               indices: list,
               remove_special_tokens: bool = True) -> str:
        """Преобразует список индексов обратно в текст.

        Args:
            indices: Список числовых индексов токенов.
            remove_special_tokens: Нужно ли удалять <pad>, <unk>, <sos>, <eos>.

        Returns:
            str: Декодированная строка с базовым форматированием пунктуации.
        """
        if remove_special_tokens:
            special_indices = [0,1,2,3] # индексы спец токенов
            indices = [idx for idx in indices if idx not in special_indices]

        words = [self.idx2word.get(idx, '<unk>') for idx in indices]

        decoded = ""
        for word in words:
            if word.endswith("@@"):
                decoded += word[:-2]
            else:
                decoded += word + " "
        return decoded.strip()


class TranslationDataset(Dataset):
    def __init__(self,
                 src_file='/content/train_split.bpe.en', # путь к BPE-файлу с исходным языком
                 tgt_file='/content/train_split.bpe.ru', # путь к BPE-файлу с целевым языком
                 src_vocab=None,          # объект Vocabulary для английского
                 tgt_vocab=None,          # объект Vocabulary для русского
                 max_length=100):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length - 2
        self.src_sentences = []
        self.tgt_sentences = []

        src = open(src_file, 'r')
        tgt = open(tgt_file, 'r')
        for s, t in zip(src, tgt):
            s_tokens = s.strip().split()
            t_tokens = t.strip().split()

            if len(s_tokens) > self.max_length:
                s_tokens = s_tokens[:self.max_length]
            if len(t_tokens) > self.max_length:
                t_tokens = t_tokens[:self.max_length]

            self.src_sentences.append(s_tokens)
            self.tgt_sentences.append(t_tokens)

        src.close()
        tgt.close()


    def __len__(self) -> int:
        return len(self.src_sentences)


    def __getitem__(self, idx: int):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        return {'src': torch.tensor(self.src_vocab.encode_tokens(src)),
                'tgt': torch.tensor(self.tgt_vocab.encode_tokens(tgt))}


    @staticmethod
    def collate_fn(batch):
        src = [item['src'] for item in batch]
        tgt = [item['tgt'] for item in batch]

        src_padded = pad_sequence(src, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(tgt, batch_first=True, padding_value=0)

        decoder_input = tgt_padded[:, :-1]   # убираем последний <eos>
        decoder_target = tgt_padded[:, 1:]   # убираем первый <sos>

        # Создаем маски для игнорирования <pad> при вычислении loss
        src_mask = (src_padded != 0)   # True для реальных токенов, False для <pad>


        return {
            'src': src_padded,               # [batch, src_len]
            'decoder_input': decoder_input,  # [batch, tgt_len - 1]
            'decoder_target': decoder_target,# [batch, tgt_len - 1]
            'src_mask': src_mask,            # [batch, src_len]
        }

    @staticmethod
    def create_translation_dataloader(dataset,
                                  batch_size: int = 32,
                                  shuffle: bool = True,
                                  num_workers: int = 0) -> DataLoader:
        """
        Создает DataLoader для TranslationDataset.

        Args:
            dataset: экземпляр TranslationDataset
            batch_size: размер батча
            shuffle: перемешивать ли данные
            num_workers: количество процессов (в Colab ставьте 0)

        Returns:
            DataLoader: настроенный загрузчик данных
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=TranslationDataset.collate_fn,
            pin_memory=True
        )


def attention_train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)                        # [batch, src_len]
        decoder_input  = batch['decoder_input'].to(device)   # [batch, tgt_len - 1]
        decoder_target  = batch['decoder_target'].to(device) # [batch, tgt_len - 1]
        src_mask = batch['src_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(src, decoder_input, src_mask=src_mask) # (batch_size, tgt_len, tgt_vocab_size)
        outputs = outputs.reshape(-1, outputs.shape[-1])

        decoder_target_flat = decoder_target.reshape(-1)

        loss = criterion(outputs, decoder_target_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def attention_evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)                        # [batch, src_len]
            decoder_input  = batch['decoder_input'].to(device)   # [batch, tgt_len - 1]
            decoder_target  = batch['decoder_target'].reshape(-1).to(device) # [batch, tgt_len - 1]
            src_mask = batch['src_mask'].to(device)


            outputs = model(src, decoder_input, teacher_forcing_ratio=0.0, src_mask=src_mask) # (batch_size, tgt_len, tgt_vocab_size)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            loss = criterion(outputs, decoder_target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def attention_greedy_decode(model, src, max_len, device, tgt_vocab, src_mask=None):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        decoder_input = torch.tensor([[tgt_vocab.word2idx['<sos>']]]).to(device)
        decoded_tokens = []
        for _ in range(max_len):
            logits, hidden, cell, attn_weights = model.decoder(decoder_input, hidden, cell, encoder_outputs, src_mask)
            next_token = logits.argmax(dim=-1)

            if next_token.item() == tgt_vocab.word2idx['<eos>']:
                break

            decoded_tokens.append(next_token.item())
            decoder_input = next_token.unsqueeze(1)
        return decoded_tokens


def attention_calculate_bleu(model, dataloader, device, tgt_vocab, max_len):
    model.eval()
    references = []
    hypotheses = []

    smoothing = SmoothingFunction().method1
    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['decoder_target'].cpu()
            src_mask = batch['src_mask'].to(device)

            for i in range(src.shape[0]):
                src_single = src[i].unsqueeze(0)
                mask_single = src_mask[i].unsqueeze(0)

                decoded_indices = attention_greedy_decode(model, src_single, max_len,device, tgt_vocab, src_mask=mask_single)

                hypothesis = tgt_vocab.decode(decoded_indices, remove_special_tokens=True)
                hypotheses.append(hypothesis.split())

                tgt_single = tgt[i]
                tgt_single = tgt_single[tgt_single != 0]
                reference = tgt_vocab.decode(tgt_single.tolist(), remove_special_tokens=True)
                references.append([reference.split()])

    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    return bleu_score


def attention_train_translator(model,
                               epochs,
                               train_loader,
                               val_loader,
                               optimizer,
                               criterion,
                               device,
                               tgt_vocab,
                               max_len,
                               scheduler=None):
    train_losses = []
    val_losses = []
    bleu_scores = []

    for epoch in range(epochs):
        # Обучение
        train_loss = attention_train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Валидация
        val_loss = attention_evaluate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss)

        # Расчет BLEU каждые 5 эпох
        if epoch % 5 == 0:
            bleu = attention_calculate_bleu(model, val_loader, device, tgt_vocab, max_len)
            bleu_scores.append(bleu)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, BLEU = {bleu:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    return train_losses, val_losses, bleu_scores
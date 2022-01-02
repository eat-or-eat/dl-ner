config = {
    # 路径系列参数
    'vocab_path': './data/vocab.txt',
    'train_path': './data/train/',
    'test_path': './data/test/',
    'schema_path': './data/schema.json',
    'model_path': './output/model/',
    # 模型系列参数
    'max_length': 150,
    'hidden_size': 128,
    'batch_size': 512,
    'epoch': 20,
    'model_type': 'lstm',
    'use_crf': True,
    'train': True,
    # 优化器系列参数
    'optimizer': 'adam',
    'learning_rate': 1e-3,
}

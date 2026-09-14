"""Compact training exports; internal canonical records stay lossless."""
from copy import deepcopy

OUTPUT_KEYS = (
    'qa_id', 'source_qa_id', 'id', 'chunk_index', 'question', 'answer',
    'thinking', 'reasoning_effort', 'eval', 'messages',
    'chat_template_kwargs', 'thinking_tokens',
    'thinking_generator', 'source_metadata',
)


def output_family(record):
    qa_id = record.get('qa_id')
    if not isinstance(qa_id, str):
        raise ValueError('Missing output qa_id')
    key, separator, family = qa_id.rpartition(':')
    if not separator or not key or family not in ('qwen3_8', 'llm_jp_4'):
        raise ValueError('Unrecognized output qa_id/family')
    return key, family


def compact_output(record):
    _, family = output_family(record)
    if record.get('target_model_family', family) != family:
        raise ValueError('qa_id and target_model_family disagree')
    for key in ('source_qa_id', 'question', 'answer', 'thinking', 'reasoning_effort',
                'messages', 'thinking_tokens', 'thinking_generator'):
        if key not in record or record[key] is None:
            raise ValueError(f'Missing required output field: {key}')
    efforts = ('low', 'medium', 'xhigh') if family == 'qwen3_8' else ('low', 'medium', 'high')
    if record['reasoning_effort'] not in efforts:
        raise ValueError('Unexpected model reasoning_effort')
    if family == 'llm_jp_4':
        kwargs = record.get('chat_template_kwargs')
        if (not isinstance(kwargs, dict)
                or set(kwargs) != {'reasoning_effort', 'conversation_start_date'}
                or kwargs['reasoning_effort'] != record['reasoning_effort']
                or not isinstance(kwargs['conversation_start_date'], str)
                or not kwargs['conversation_start_date']):
            raise ValueError('Invalid llm-jp chat_template_kwargs')
    messages = record['messages']
    thinking_field = 'reasoning_content' if family == 'qwen3_8' else 'thinking'
    expected = [
        {'role': 'user', 'content': record['question']},
        {'role': 'assistant', 'content': record['answer'], thinking_field: record['thinking']},
    ]
    if messages != expected:
        raise ValueError('Messages do not match question/answer/thinking or model schema')
    metadata = record.get('source_metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Missing original source_metadata')
    result = {key: deepcopy(record.get(key)) for key in OUTPUT_KEYS
              if key != 'chat_template_kwargs' or family == 'llm_jp_4'}
    result['source_metadata'] = {key: deepcopy(metadata.get(key))
                                 for key in ('qa_id', 'id', 'chunk_index')}
    return result

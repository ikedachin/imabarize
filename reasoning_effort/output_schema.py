"""Compact training exports; internal canonical records stay lossless."""
from copy import deepcopy

OUTPUT_KEYS = (
    'qa_id', 'question', 'answer',
    'thinking', 'reasoning_effort', 'eval', 'messages',
    'chat_template_kwargs', 'thinking_tokens',
    'thinking_generator', 'source_metadata',
)


def llmjp_messages(record):
    if record.get('reasoning_effort') not in ('low', 'medium', 'high'):
        raise ValueError('Unexpected llm-jp reasoning_effort')
    for field in ('question', 'thinking', 'answer'):
        if not isinstance(record.get(field), str):
            raise ValueError(f'Invalid llm-jp {field}')
    return [
        {'role': 'user', 'content': record['question']},
        {'role': 'assistant', 'content': record['answer'], 'thinking': record['thinking']},
    ]


def validate_llmjp_messages(record):
    if record.get('messages') != llmjp_messages(record):
        raise ValueError('Invalid llm-jp messages')


def template_kwargs(family, effort):
    if family == 'llm_jp_4':
        return {'reasoning_effort': effort, 'conversation_start_date': '2026-09-11'}
    return {'reasoning_effort': effort, 'enable_thinking': True, 'preserve_thinking': True}


def output_family(record):
    qa_id = record.get('qa_id')
    if qa_id is None:
        from reasoning_effort.cache import stable_id
        # Legacy exports omitted qa_id. The
        # content hash connects the exact output to its canonical resume entry.
        messages = record.get('messages', [])
        if any(isinstance(m, dict) and 'reasoning_content' in m for m in messages):
            family = 'qwen3_8'
        elif any(isinstance(m, dict) and ('thinking' in m or m.get('channel') in
                 ('analysis_imabari', 'final_imabari', 'analysis', 'final')) for m in messages):
            family = 'llm_jp_4'
        else:
            raise ValueError('Unrecognized legacy output model family')
        return stable_id(record), family
    if not isinstance(qa_id, str):
        raise ValueError('Invalid output qa_id')
    key, separator, family = qa_id.rpartition(':')
    if not separator or not key or family not in ('qwen3_8', 'llm_jp_4'):
        raise ValueError('Unrecognized output qa_id/family')
    return key, family


def compact_output(record):
    _, family = output_family(record)
    if record.get('target_model_family', family) != family:
        raise ValueError('qa_id and target_model_family disagree')
    for key in ('qa_id', 'question', 'answer', 'thinking', 'reasoning_effort',
                'messages', 'thinking_tokens', 'thinking_generator'):
        if key not in record or record[key] is None:
            raise ValueError(f'Missing required output field: {key}')
    efforts = ('low', 'medium', 'xhigh') if family == 'qwen3_8' else ('low', 'medium', 'high')
    if record['reasoning_effort'] not in efforts:
        raise ValueError('Unexpected model reasoning_effort')
    kwargs = record.get('chat_template_kwargs')
    expected_kwargs = template_kwargs(family, record['reasoning_effort'])
    if kwargs != expected_kwargs or (family == 'qwen3_8' and
            (kwargs.get('enable_thinking') is not True or kwargs.get('preserve_thinking') is not True)):
        raise ValueError('Invalid chat_template_kwargs')
    if family == 'llm_jp_4':
        validate_llmjp_messages(record)
    else:
        expected = [
            {'role': 'user', 'content': record['question']},
            {'role': 'assistant', 'content': record['answer'], 'reasoning_content': record['thinking']},
        ]
        if record['messages'] != expected:
            raise ValueError('Messages do not match question/answer/thinking or model schema')
    metadata = record.get('source_metadata')
    if not isinstance(metadata, dict):
        raise ValueError('Missing original source_metadata')
    result = {key: deepcopy(record.get(key)) for key in OUTPUT_KEYS}
    result['source_metadata'] = {key: deepcopy(metadata.get(key))
                                 for key in ('qa_id', 'id', 'chunk_index')}
    return result

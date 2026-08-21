from src.agent_tools import parse_tool_blocks, strip_tool_blocks


def test_gemma_tool_call_json_args_parse_and_strip():
    raw = '<|tool_call|>call:web_search{"query":"hello world"}<|tool_call|>'

    blocks = parse_tool_blocks(raw)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert blocks[0].content == "hello world"
    assert strip_tool_blocks(raw).strip() == ""


def test_gemma_tool_call_unquoted_args_parse():
    raw = '<|tool_call|>call:web_search{query: "hello world"}<|tool_call|>'

    blocks = parse_tool_blocks(raw)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "web_search"
    assert blocks[0].content == "hello world"


def test_gemma_tool_call_normalizes_dash_tool_name():
    raw = '<|tool_call|>call:read-file{"path":"README.md"}<|tool_call|>'

    blocks = parse_tool_blocks(raw)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "read_file"
    assert blocks[0].content == "README.md"


def test_gemma_parser_does_not_strip_non_tool_fenced_metadata():
    raw = '```python id="abc"\nprint("hello")\n```'

    assert parse_tool_blocks(raw) == []
    assert strip_tool_blocks(raw) == raw


def test_strip_raw_openai_function_json_leak():
    raw = (
        '{"function":{"arguments":"{\\"action\\":\\"search\\",\\"tex\\":\\"hi\\"}",'
        '"name":"manage_memory"},"id":"call_memory_search1","type":"function"}]</|assistan|Done.'
    )

    assert strip_tool_blocks(raw) == "Done."


def test_strip_raw_openai_function_json_array_leak():
    raw = (
        'Before\n['
        '{"function":{"arguments":"{\\"action\\":\\"add\\",\\"text\\":\\"x\\"}",'
        '"name":"manage_memory"},"id":"call_memory_add1","type":"function"}'
        ']\nAfter'
    )

    assert strip_tool_blocks(raw) == "Before\nAfter"


def test_gemma_tool_call_mcp_tools_namespace():
    raw = '<|tool_call>call:mcp_tools:manage_calendar{end_time: "2026-09-17T18:00:00", location: "베트남", start_time: "2026-09-13T09:00:00", summary: "사장님 베트남 출장"}<tool_call|>'

    blocks = parse_tool_blocks(raw)

    assert len(blocks) == 1
    assert blocks[0].tool_type == "manage_calendar"
    import json
    parsed = json.loads(blocks[0].content)
    assert parsed["summary"] == "사장님 베트남 출장"
    assert parsed["action"] == "create_event"
    assert strip_tool_blocks(raw).strip() == ""


def test_gemma_tool_call_email_limit_and_aliases():
    raw = '<|tool_call>call:mcp_tools:list_emails{limit:5}<tool_call|>'
    blocks = parse_tool_blocks(raw)
    assert len(blocks) == 1
    assert blocks[0].tool_type == "mcp__email__list_emails"
    import json
    parsed = json.loads(blocks[0].content)
    assert parsed["max_results"] == 5
    assert strip_tool_blocks(raw).strip() == ""

    raw_alias = '<|tool_call|>call:get_latest_emails{"limit": 3}<|tool_call|>'
    blocks_alias = parse_tool_blocks(raw_alias)
    assert len(blocks_alias) == 1
    assert blocks_alias[0].tool_type == "mcp__email__list_emails"
    parsed_alias = json.loads(blocks_alias[0].content)
    assert parsed_alias["max_results"] == 3


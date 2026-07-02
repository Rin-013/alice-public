"""Sanity tests for the streaming tool-call parser."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from streaming.twitch.tool_call_parser import StreamingToolCallParser


def feed_chunks(chunks):
    p = StreamingToolCallParser()
    visible_total = ""
    calls = []
    for c in chunks:
        v, cs = p.feed(c)
        visible_total += v
        calls.extend(cs)
    tail, more = p.flush()
    visible_total += tail
    calls.extend(more)
    return visible_total, calls


def test_no_tool_call():
    v, c = feed_chunks(["hello ", "world"])
    assert v == "hello world", v
    assert c == [], c
    print("PASS: no_tool_call")


def test_single_clean():
    call_body = (
        "\n<function=timeout_user>\n"
        "<parameter=username>spammer</parameter>\n"
        "<parameter=duration_seconds>60</parameter>\n"
        "<parameter=reason>spamming</parameter>\n"
        "</function>\n"
    )
    v, c = feed_chunks([
        "okay you're done. ",
        f'<tool_call>{call_body}</tool_call>',
        ' five minutes.',
    ])
    assert v == "okay you're done.  five minutes.", repr(v)
    assert len(c) == 1
    assert c[0].tool == "timeout_user"
    assert c[0].args["username"] == "spammer"
    assert c[0].args["duration_seconds"] == 60
    assert c[0].parse_error is None
    print("PASS: single_clean")


def test_split_open_tag():
    call_body = "<function=read_recent_chat>\n</function>"
    chunks = ['hi ', '<', 'tool_', 'call', f'>{call_body}</tool_call> done']
    v, c = feed_chunks(chunks)
    assert v == "hi  done", repr(v)
    assert len(c) == 1 and c[0].tool == "read_recent_chat"
    print("PASS: split_open_tag")


def test_split_close_tag():
    call_body = (
        "<function=ban_user>\n"
        "<parameter=username>x</parameter>\n"
        "<parameter=reason>y</parameter>\n"
        "</function>"
    )
    chunks = [
        f'<tool_call>{call_body}',
        '</',
        'tool_',
        'call>',
        ' bye',
    ]
    v, c = feed_chunks(chunks)
    assert v == " bye", repr(v)
    assert len(c) == 1 and c[0].tool == "ban_user"
    print("PASS: split_close_tag")


def test_partial_lookalike():
    chunks = ["look at this <tool", "box", " over there"]
    v, c = feed_chunks(chunks)
    assert v == "look at this <toolbox over there", repr(v)
    assert c == []
    print("PASS: partial_lookalike")


def test_bad_format():
    chunks = ['<tool_call>some garbage with no function tag</tool_call>']
    v, c = feed_chunks(chunks)
    assert v == "", repr(v)
    assert len(c) == 1
    assert c[0].parse_error and "missing" in c[0].parse_error
    print("PASS: bad_format")


def test_unclosed_tag():
    chunks = ['talking <tool_call><function=foo>\n<parameter=x>1</parameter>', '\n</function>']
    v, c = feed_chunks(chunks)
    assert v == "talking ", repr(v)
    assert len(c) == 1
    assert c[0].parse_error and "unclosed" in c[0].parse_error
    print("PASS: unclosed_tag")


def test_multiple_calls():
    call_a = "<function=a>\n</function>"
    call_b = "<function=b>\n</function>"
    chunks = [
        'first ',
        f'<tool_call>{call_a}</tool_call>',
        ' middle ',
        f'<tool_call>{call_b}</tool_call>',
        ' end',
    ]
    v, c = feed_chunks(chunks)
    assert v == "first  middle  end", repr(v)
    assert len(c) == 2
    assert c[0].tool == "a" and c[1].tool == "b"
    print("PASS: multiple_calls")


def test_token_per_char():
    msg = '<tool_call><function=x>\n</function></tool_call>'
    chunks = ['hi '] + list(msg) + [' bye']
    v, c = feed_chunks(chunks)
    assert v == "hi  bye", repr(v)
    assert len(c) == 1 and c[0].tool == "x"
    print("PASS: token_per_char")


def test_no_params():
    chunks = ['<tool_call><function=get_sub_count>\n</function></tool_call>']
    v, c = feed_chunks(chunks)
    assert v == ""
    assert len(c) == 1
    assert c[0].tool == "get_sub_count"
    assert c[0].args == {}
    assert c[0].parse_error is None
    print("PASS: no_params")


def test_json_param_value():
    call_body = (
        '<function=create_poll>\n'
        '<parameter=title>Best game?</parameter>\n'
        '<parameter=choices>["Elden Ring", "Hollow Knight"]</parameter>\n'
        '</function>'
    )
    v, c = feed_chunks([f'<tool_call>{call_body}</tool_call>'])
    assert len(c) == 1
    assert c[0].tool == "create_poll"
    assert c[0].args["title"] == "Best game?"
    assert c[0].args["choices"] == ["Elden Ring", "Hollow Knight"]
    print("PASS: json_param_value")


def test_multiline_param():
    call_body = (
        '<function=mind_think>\n'
        '<parameter=topic>\nwhy chat is quiet and how to break the silence\n</parameter>\n'
        '</function>'
    )
    v, c = feed_chunks([f'<tool_call>{call_body}</tool_call>'])
    assert len(c) == 1
    assert c[0].tool == "mind_think"
    assert "quiet" in c[0].args["topic"]
    print("PASS: multiline_param")


if __name__ == "__main__":
    test_no_tool_call()
    test_single_clean()
    test_split_open_tag()
    test_split_close_tag()
    test_partial_lookalike()
    test_bad_format()
    test_unclosed_tag()
    test_multiple_calls()
    test_token_per_char()
    test_no_params()
    test_json_param_value()
    test_multiline_param()
    print("\nAll tool-call parser tests passed.")

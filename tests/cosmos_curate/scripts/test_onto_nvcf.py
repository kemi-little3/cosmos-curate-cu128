from cosmos_curate.scripts.onto_nvcf import _select_node_ip


def test_select_node_ip_prefers_non_loopback_when_hostname_returns_multiple_ips() -> None:
    assert _select_node_ip("127.0.0.1 192.18.5.222") == "192.18.5.222"


def test_select_node_ip_falls_back_to_first_loopback_ip() -> None:
    assert _select_node_ip("127.0.0.1") == "127.0.0.1"

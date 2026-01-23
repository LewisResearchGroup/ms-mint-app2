from unittest.mock import MagicMock, patch

from ms_mint_app.plugins.target_optimization import _toggle_bookmark_logic


def test_toggle_bookmark_logic_false_to_true():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [False]

    with patch("ms_mint_app.plugins.target_optimization.duckdb_connection") as mock_conn_mgr, \
            patch("ms_mint_app.plugins.target_optimization.fac.AntdIcon") as mock_icon:
        mock_conn_mgr.return_value.__enter__.return_value = mock_conn

        _toggle_bookmark_logic("TestTarget", "wd")

        calls = mock_conn.execute.call_args_list
        assert len(calls) >= 2
        update_call = calls[-1]
        args, _ = update_call
        assert "UPDATE targets SET bookmark = ?" in args[0]
        assert args[1] == [True, "TestTarget"]
        mock_icon.assert_called_with(icon="antd-star", style={"color": "gold"})


def test_toggle_bookmark_logic_true_to_false():
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = [True]

    with patch("ms_mint_app.plugins.target_optimization.duckdb_connection") as mock_conn_mgr, \
            patch("ms_mint_app.plugins.target_optimization.fac.AntdIcon") as mock_icon:
        mock_conn_mgr.return_value.__enter__.return_value = mock_conn

        _toggle_bookmark_logic("TestTarget", "wd")

        calls = mock_conn.execute.call_args_list
        update_call = calls[-1]
        args, _ = update_call
        assert args[1] == [False, "TestTarget"]
        mock_icon.assert_called_with(icon="antd-star", style={"color": "gray"})

from compare_methods import compare, METHODS

def test_comparison_covers_all_methods():
    result=compare()
    assert list(result["summary"].index)==METHODS
    assert int(result["summary"].loc["Isolation Forest","Flagged"])==32
    assert result["stability"]["Mean Jaccard"].between(0,1).all()
    assert len(result["flags"])==260

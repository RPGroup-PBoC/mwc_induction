"""
Title:
    mwc_induction_utils_test.py
Author(s):
    Griffin Chure
Creation Date:
    2016-10-14
Purpose:
    Test suite for MWC induction utilities.
License:
    MIT. See `mwc_induction_utils.py` for detailed licensing information.
"""
import numpy as np
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
import pytest
import pandas as pd
import imp

imp.reload(mwc)

# ###################
# General Thermodynamic Functions
# ###################


# ###################
def test_pact_log():
    log_val = np.log(2)
    test_iptg = np.array([1, 1, 1])
    assert (mwc.pact_log(test_iptg, 0, 0, 0) == 0.5).all()
    assert (mwc.pact_log(test_iptg, log_val, log_val, -log_val) == 1/3).all()
    with pytest.raises(ValueError):
        neg_iptg = np.array([1, -1, 1])
        mwc.pact_log(bad_iptg, 0, 0, 0)


# ###################
def test_fold_change_log():
    test_iptg = np.array([1, 1, 1])
    test_num_rep = 1
    large_num_rep = 4.6E6
    assert np.isclose(mwc.fold_change_log(test_iptg, 0, 0, 0,
                                          test_num_rep, 0), 1).all()
    assert np.isclose(mwc.fold_change_log(test_iptg, 0, 0, 0,
                                          large_num_rep, 0,
                                          quaternary_state=2), 1/3).all()
    assert np.isclose(mwc.fold_change_log(test_iptg, 0, 0, 0,
                                          large_num_rep, 0,
                                          quaternary_state=1), 1/2).all()
    with pytest.raises(ValueError):
        neg_iptg = np.array([1, -1, 1])
        neg_num_repressors = -1
        bad_quat_state = 0
        mwc.fold_change_log(neg_iptg, 0, 0, 0, test_num_rep, 0)
        mwc.fold_change_log(test_iptg, 0, 0, 0, neg_num_rep, 0)
        mwc.fold_change_log(test_iptg, 0, 0, 0, test_num_rep, 0,
                            quaternary_state=bad_quat_state)
        mwc.fold_chagne_log(test_iptg, 0, 0, 0, test_num_rep, 0,
                            nonspec_sites=-1)


# ###################
def test_fold_change_log_rnap():
    test_iptg = np.array([1, 1, 1])
    test_num_rep = 1
    test_num_pol = 1
    large_num_rep = 4.6E6
    large_num_pol = 4.6E6
    assert np.isclose(mwc.fold_change_log_rnap(test_iptg, 0, 0, 0,
                                               test_num_rep, 0, test_num_pol,
                                               0), 1).all()

    assert (mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, large_num_rep, 0,
                                     large_num_pol, 0) == 1/2).all()

    assert (mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, large_num_rep, 0,
                                     large_num_pol, 0, quaternary_state=1) ==
            2/3).all()

    with pytest.raises(ValueError):
        neg_iptg = np.array([1, -1, 1])
        neg_num_rep = -1
        neg_num_pol = -1
        bad_quat_state = 0
        bad_nonspec_sites = 0
        mwc.fold_change_log_rnap(neg_iptg, 0, 0, 0, test_num_rep, 0,
                                 test_num_pol, 0)
        mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, neg_num_rep, 0,
                                 test_num_pol, 0)
        mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, test_num_rep, 0,
                                 neg_num_pol, 0)
        mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, test_num_rep, 0,
                                 test_num_pol, 0,
                                 quaternary_state=bad_quat_state)
        mwc.fold_change_log_rnap(test_iptg, 0, 0, 0, test_num_rep, 0,
                                 test_num_pol, 0,
                                 nonspec_sites=bad_nonspec_sites)


# Need to test range of epsilon values here.
# ###################
def test_bohr_fn():
    columns = ('IPTG_uM', 'repressors', 'binding_energy')
    test_df = pd.DataFrame(3 * [[1, 4.6E6, 0]], columns=columns)
    assert (np.exp(mwc.bohr_fn(test_df, 0, 0, epsilon=0)) == 1/2).all()
    with pytest.raises(RuntimeError):
        bad_df = pd.DataFrame([])
        mwc.bohr_fn(bad_df, 0, 0, 0,)
    with pytest.raises(ValueError):
        mwc.bohr_fn(test_df, 0, 0, quaternary_state=-1)
        mwc.bohr_fn(test_df, 0, 0, nonspec_sites=0)

# ###################
# General Thermodynamic Functions
# ###################


# ###################
def test_log_post():
    zero_params = np.array([0, 0])
    val_params = np.array([np.log(2), np.log(2)])
    indep_var = np.array([1, 4.6E6, 0])
    dep_var = np.array([1, 1, 1])
    with pytest.raises(RuntimeError):
        mwc.log_post(zero_params, np.array([1]), dep_var)
        mwc.log_post(zero_params, indep_var, np.array([1, 1]))
    with pytest.raises(ValueError):
        mwc.log_post(zero_params, indep_var, dep_var,
                     quaternary_state=-1)
        mwc.log_post(zero_params, indep_var, dep_var,
                     nonspec_sites=0)


test_pact_log()
test_fold_change_log()
test_fold_change_log_rnap()
test_bohr_fn()
test_log_post()

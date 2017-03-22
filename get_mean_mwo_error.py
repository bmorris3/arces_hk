from toolkit import get_duncan_catalog

table = get_duncan_catalog()
errors = table['e_Smean'].data.data
nonzero_errors = errors[errors != 0]

print('Mean error on an MWO S-index measurement: {}'
      .format(nonzero_errors.mean()))
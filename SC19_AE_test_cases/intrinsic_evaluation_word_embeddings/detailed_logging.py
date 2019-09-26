def logPredictions(dataset, predictions, gold, dsname, stream):
    unified = []
    for i in range(len(dataset)):
        unified.append([dataset[i], predictions[i], gold[i], 0, 0, 0])

    max_left, max_right = 0, 0
    for (left, right, _) in dataset:
        if len(left) > max_left: max_left = len(left)
        if len(right) > max_right: max_right = len(right)

    # get gold indexing
    unified.sort(key=lambda k:k[2], reverse=True)
    for i in range(len(unified)):
        unified[i][4] = i
    # get pred indexing
    unified.sort(key=lambda k:k[1], reverse=True)
    for i in range(len(unified)):
        unified[i][3] = i
    # get errors
    for i in range(len(unified)):
        unified[i][5] = abs(unified[i][4] - unified[i][3])

    # write in gold order
    unified.sort(key=lambda k:k[2], reverse=True)
    stream.write('\n\n== %s :: Gold order =============================================================================================\n' % dsname)
    for ((left, right, _), pred, gold, pred_ix, gold_ix, error) in unified:
        stream.write(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]\n').format(max_left, max_right)
            % (left, right, pred, pred_ix, gold, gold_ix, error))

    # write in prediction order
    unified.sort(key=lambda k:k[1], reverse=True)
    stream.write('\n\n== %s :: Predicted order ========================================================================================\n' % dsname)
    for ((left, right, _), pred, gold, pred_ix, gold_ix, error) in unified:
        stream.write(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]\n').format(max_left, max_right)
            % (left, right, pred, pred_ix, gold, gold_ix, error))

    # get top 10 best and worst errors
    unified.sort(key=lambda k:k[5], reverse=True)
    stream.write('\n\n== %s :: Worst errors ===========================================================================================\n' % dsname)
    for ((left, right, _), pred, gold, pred_ix, gold_ix, error) in unified[:10]:
        stream.write(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]\n').format(max_left, max_right)
            % (left, right, pred, pred_ix, gold, gold_ix, error))
    unified.sort(key=lambda k:k[5], reverse=False)
    stream.write('\n\n== %s :: Best errors ============================================================================================\n' % dsname)
    for ((left, right, _), pred, gold, pred_ix, gold_ix, error) in unified[:10]:
        stream.write(('%{0}s  %{1}s  -->  %f (%3d)  ||  %f (%3d)  [Error: %d]\n').format(max_left, max_right)
            % (left, right, pred, pred_ix, gold, gold_ix, error))

    stream.write('\n')

from darknet import Darknet

net = (
    'cfg/jav.data',
    'cfg/jav.cfg',
    'backup/jav.weights'
)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: %s file' % sys.argv[0])
        sys.exit(1)

    fn = sys.argv[1]

    from os import walk
    from time import perf_counter
    from os.path import isdir, exists, join

    filenames = []

    if isdir(fn):
        dp, _, fn = next(walk(fn))
        filenames.extend(join(dp, f) for f in fn)
    elif exists(fn):
        filenames.append(fn)

    dn = Darknet(*net)
    total = len(filenames)
    mag = len(str(total))

    sn = 1
    start = perf_counter()

    for fn in filenames:
        t_fn = fn.split('.')[0]
        l_fn = t_fn + '.txt'

        if exists(l_fn):
            continue

        cinfo = '{:{pad}d}/{}'.format(sn, total, pad=mag)
        t = perf_counter()
        with open(l_fn, 'w') as lb:
            for c, i, b in sorted(dn(fn), reverse=True):
                # print('\t%s (%.3f)' % (dn.name(i), c))
                print('%.2f %d %s' % (c, i, ' '.join('%.6f' % p for p in b)), file=lb)
        now = perf_counter()
        tinfo = '%2.3fs (%.1f/s)' % (now - t, sn / (now - start))
        print('\r%-12s| %-60s| %-10s' % (cinfo, t_fn, tinfo), end='')
        sn += 1

    print('\nDone')

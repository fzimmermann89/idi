from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, line_cell_magic
import argparse
import shlex
import subprocess
import os
import time
import subprocess
import getpass
import IPython.display

'''
IPython magic to submit jobs
'''


@magics_class
class submit_magics(Magics):
    defaultline = ''

    @classmethod
    def _run(cls, defaultcmd, key, mapping, vs, line, cell, defaultline=''):
        parser = argparse.ArgumentParser(description='Submit a job', prog=defaultcmd)
        parser.add_argument('-c', dest='c', default='4', type=str, help='n cpu per task (4)')
        parser.add_argument('-m', dest='m', default='4G', type=str, help='ram (4G)')
        parser.add_argument('-t', dest='t', default='06:00:00', type=str, help='time (6h)')
        parser.add_argument('-n', dest='n', default='submitmagic', type=str, help='name')
        parser.add_argument('-o', dest='o', default='~/joblog_%j.log', type=lambda x: os.path.expanduser(x), help="output (~/joblog_%%j.log)")
        parser.add_argument('-g', dest='g', default=None, type=str, help='gpus (None)')
        parser.add_argument('-a', dest='a', default=None, type=str, help='array')
        parser.add_argument('--interpreter', default='python', type=str, help='interpreter (python)')
        parser.add_argument('--batch_cmd', metavar='cmd', type=str, default=defaultcmd, nargs='*', help='batch command')
        parser.add_argument('--dryrun', action='store_true', help="don't submit")
        parser.add_argument('--debug', action='store_true', help="print debug info")
        parser.add_argument('--no-replace', dest='replace', action='store_false', help="don't replace scheduler variables with values")
        parser.epilog = f'defaults can be set with %submitdefault line magic. {"currently set:" if defaultline else ""} {defaultline}'

        try:
            args, u = parser.parse_known_args(shlex.split(' '.join([line, defaultline])))
        except SystemExit:
            return

        unknown = []
        for el in u:
            if el[0] == '-':
                unknown.append(el)
            elif len(unknown) > 0:
                unknown[-1] += ' ' + el
        if len(vs) and args.replace:
            prepcmd = 'sed ' + ' '.join([f'-e "s^{{{{{v}}}}}^${v}^g"' for v in vs])
        else:
            prepcmd = 'cat'
        cmd = '\n'.join(
            ['#!/bin/sh']
            + [f'{key} {mapping[k]}={v}' for k, v in vars(args).items() if (v is not None and k in mapping)]
            + [f'{key} {v}' for v in unknown]
            + (['', 'hostname', 'env', 'pwd', 'ls -haltr /scratch/', 'mount'] if args.debug else [])
            + [f'{prepcmd} << "EOF_PYTHONFILE" | {args.interpreter}', '', cell, '', 'EOF_PYTHONFILE']
        )
        if args.debug or args.dryrun:
            print(args.batch_cmd)
            print(cmd)
        if not args.dryrun:
            try:
                p = subprocess.run(args.batch_cmd, input=cmd.encode('utf-8'), capture_output=True)
            except FileNotFoundError:
                return 'Submit Command not found'
            if p.returncode or args.debug:
                    print(p.stderr.decode('utf-8'))
                    print(p.stdout.decode('utf-8'))
            return p.returncode

    @line_magic
    def submitdefault(self, line):
        if line:
            self.defaultline = self.defaultline + ' ' + line
        else:
            print(
                f'''
usage: %submitdefault -argument val
use reload_ext to reset.
currently set: 
{self.defaultline}'''
            )

    @cell_magic
    def slurm(self, line, cell=None):
        '''       
        usage: %%slurm [-h] [-c C] [-m M] [-t T] [-n N] [-o O] [-g G] [-a A] [-i I]
        [--dryrun] [--debug] [--no-replace]
        [sbatch_cmd [sbatch_cmd ...]]

        positional arguments:
        sbatch_cmd    batch command, e.g sbatch (default), ssh loginserver sbatch, etc.

        optional arguments:
        -h, --help    show a help message
        -c C          n cpu per task (4)
        -m M          ram (4G)
        -t T          time (6h)
        -n N          name
        -o O          output (~/joblog_%j.log)
        -g G          gpus (None)
        -a A          array
        -i I          interpreter (python)
        --dryrun      don't submit
        --debug       print debug info
        --no-replace  don't replace scheduler variables with values
        '''
        defaultcmd = 'sbatch'
        mapping = {
            'c': '--cpus-per-task',
            'm': '--mem',
            't': '--time',
            'n': '--job-name',
            'o': '--output',
            'g': '--gpus',
            'a': '--array',
        }
        key = '#SBATCH'
        vs = [
            'SLURM_JOBID',
            'SLURM_ARRAY_TASK_MAX',
            'SLURM_ARRAY_TASK_MIN',
            'SLURM_ARRAY_TASK_COUNT',
            'SLURM_ARRAY_TASK_ID',
            'SLURM_ARRAY_TASK_STEP',
            'SLURM_ARRAY_JOB_ID',
            'SLURM_NODELIST',
            'SLURM_JOB_NAME',
            'SLURM_GPUS',
            'SLURM_JOB_CPUS_PER_NODE',
            'SLURM_MEM_PER_NODE',
        ]
        return submit_magics._run(defaultcmd, key, mapping, vs, line, cell, self.defaultline)

    @cell_magic
    def pbs(self, line, cell=None):
        raise NotImplementedError('pbs not implmented')

    @cell_magic
    def lsf(self, line, cell=None):
        raise NotImplementedError('lsf not implmented')
        
        
    @line_magic
    def queue(self, line=None):
        cmd=shlex.split(line) or ['/opt/slurm/bin/squeue','-u', getpass.getuser()]
        try:
            display=IPython.display.display({'text/plain':'getting data..'},raw=True,display_id=True);
            while True:
                p=subprocess.run(cmd,capture_output=True,timeout=5)
                if not p.returncode:
                    output=p.stdout.decode('ascii')
                    html = (
                        "<table><tr><td>"
                        + "</tr><tr><td>".join(
                            ("</td><td>".join(line.split()) for line in output.split("\n") if line)
                        )
                        + "</td></tr></table>"
                    )
                    display.update({'text/html':html+f'<i>Live</i>'},raw=True)
                    time.sleep(1)

                else: 
                    break
        except KeyboardInterrupt:
             display.update({'text/html':html+f'<i>Last Updated at {time.asctime(time.localtime())}</i>'},raw=True)


def load_ipython_extension(ipython):
    ipython.register_magics(submit_magics)

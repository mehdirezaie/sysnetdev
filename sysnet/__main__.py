#!/usr/bin/env python
""" 
    A high-level code for running the SYSNet software

    Take a look into the config file under the directory 'scripts'
    to learn about the input parameters.
    
    Mehdi Rezaie, mr095415@ohio.edu
    October 2020
"""
def main(debug=False):
    import sys
    import sysnet
    
    try:
        yaml_config = sys.argv[1].lower()
    except IndexError:  # no command
        print("Pass config file or command line arguments")
        exit()
    if yaml_config.startswith('-'):  # not a config file, but an argparse argument
        yaml_config = None
    else:
        sys.argv.pop(1)

    if debug:
        sysnet.detect_anomaly() # this will slow down

    config = sysnet.parse_cmd_arguments(yaml_config=yaml_config)
    pipeline = sysnet.SYSNet(config)
    pipeline.run()

if __name__ == '__main__':
    
    main()

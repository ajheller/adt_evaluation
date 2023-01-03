Bugs:
    optimize_dome_LF ignores elevation limits, fixed at 5pi/6 (wtf?)


Look at:
   https://spaudiopy.readthedocs.io/en/latest/index.html
   https://github.com/chris-hld/spaudiopy/

   https://realpython.com/pysimplegui-python/

   https://python-sounddevice.readthedocs.io/en/0.3.3/

FAUST:
. FAUST now builds with cmake... 
    so make sure you have a current version from macports or brew
. make help -- provides info about the all the targets available
version
. I checkout the FAUST git repo to ~/src/audio/faust-2022/
    some components are in subrepos, make will get this
. for VST on MacOS switch to faust2faustvst
. fix MYGCCFLAGS in either faust2vst or faustoptflags
    new version of faust2vst makes a fat binary for x86_64 and
    arm64, but doesn't consider arm64 compile on x86_64
. fix sprintf warnings in:
    faustvst.cpp  - used by faust2faustvst
    vst.cpp - used by faust2vst, faust2w64vst, faust2w32vst
. add "id" to metadata, used by vst.cpp (line 268)

Uninstalling Apple XCode IDE and installing the lastest Command Line Tools for Xcode
. Get command line tools from  14.2 is current as of Dec-2022
    https://developer.apple.com/download/all/?q=xcode
. Create uninstaller script at ~/bin/uninstall-xcode.sh based on
    https://onexlab-io.medium.com/uninstall-xcode-from-macos-eca1b69dc836
. Clue about deselecting Xcode (which is no longer installed)
    https://stackoverflow.com/questions/32674064/xcrun-error-active-developer-path-does-not-exist-use-xcode-select-switch
    sudo xcode-select --reset will get command line stuff running
    sudo /usr/bin/xcode-select --switch /Library/Developer/CommandLineTools
    xcode-select -print-path  # shows current setting
    be sure to do a 'rehash' in bash or it will still look for the old executables
    

# Bugs:
- optimize_dome_LF ignores elevation limits, fixed at 5pi/6 (wtf?)
- NumPy warning:  
  ```/Users/heller/Documents/adt_evaluation/optimize_dome.py:231: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  print("Using:\n", Sr.ids[~off.copy()], file=f)
  ```

# Look at:
- Streaming audio processing
    - https://spaudiopy.readthedocs.io/en/latest/index.html
    - https://github.com/chris-hld/spaudiopy/
- GUI
    - https://realpython.com/pysimplegui-python/
- PortAudio
    - https://python-sounddevice.readthedocs.io/en/0.3.3/
- Stand alone Python app
    - https://pyinstaller.org/en/stable/index.html


# FAUST:
- FAUST now builds with cmake
    - Apple Command line tools don't have cmake, get a current version from macports or brew

- make help -- provides info about the all the targets available version

- I checkout the FAUST git repo to ~/src/audio/faust-2022/
    - https://github.com/grame-cncm/faust
    - some components are in subrepos, make will get this

- for VST on MacOS switch to faust2faustvst

- fix MYGCCFLAGS in either faust2vst or faustoptflags
    - new version of faust2vst makes a fat binary for x86\_64 and arm64, but doesn't consider arm64 compile on x86_64

- fix sprintf warnings in:
    - faustvst.cpp  - used by faust2faustvst
    - vst.cpp - used by faust2vst, faust2w64vst, faust2w32vst

- add "id" to metadata, used by vst.cpp (line 268)

## Compiling Faust
Uninstalling Apple XCode IDE and installing the lastest Command Line Tools for Xcode
- Get command line tools from  
    - https://developer.apple.com/download/all/?q=xcode
    - 14.2 is current as of Dec-2022
- Create uninstaller script at ~/bin/uninstall-xcode.sh based on
    - https://onexlab-io.medium.com/uninstall-xcode-from-macos-eca1b69dc836
- Clue about deselecting Xcode (which is no longer installed)__
    -
    https://stackoverflow.com/questions/32674064/xcrun-error-active-developer-path-does-not-exist-use-xcode-select-switch
    ````sudo xcode-select --reset will get command line stuff running
    sudo /usr/bin/xcode-select --switch /Library/Developer/CommandLineTools
    xcode-select -print-path  # shows current setting
    ````
    - be sure to do a 'rehash' in bash or it will still look for the old
    executables

# Bidule
- https://www.plogue.com/bidule/latest/
- VST errors: https://www.plogue.com/bidule/help/ch05s05.html
- Windows VMs for testing
    - https://developer.microsoft.com/en-us/windows/downloads/virtual-machines/
    - On windows the Bidule logfile is in:
    - C:\Users\<user>\AppData\Roaming\Plogue\Bidule\bidule.log
	- LoadLibrary failing with Error 126
	    - googling ... this can have many causes, often missing dlls.
	    - I have used depends.exe to find these problems in the
        distant past

- For Windows, maybe I need the VC Redistributable
    - https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
	- Nope... they're already installed in the developer VM,
	- VSTs compiled with the online Faust compiler are recognized by
      Bidule, so something is wrong with the MingGW toolchain I have on the Mac

# Faust to jack with gui
- needs gtk+-2.0. fix with sudo port install gtk2
    - this trigger a massive package update that took about 30 minutes
    - clean up with sudo port unstall inactive
    

# Python
- Conda Forge: https://conda-forge.org/#about
- Intel Python: 
    

# Install

Roomer isn't in the default HACS list yet, so you add it as a **custom
repository**. If you don't use HACS, you can copy the files in by hand instead.

!!! question "Not sure whether you have HACS?"
    Look at the sidebar in Home Assistant. If there's an item called **HACS**,
    you have it, so use the **HACS** tab below. If there isn't, use the
    **Manual** tab, or set up HACS first by following the guide at
    [hacs.xyz](https://hacs.xyz/docs/use/). Both ways give you the same Roomer.

=== "HACS"

    1. In Home Assistant, open **HACS** from the sidebar.
    2. Select the **⋮** menu in the top right corner, then **Custom repositories**.
    3. Fill in the form:
        - **Repository:** `https://github.com/owens-ben/roomer`
        - **Type** (or **Category**): **Integration**
    4. Select **Add**, then close the dialog.

        !!! example "Screenshot needed: hacs-custom-repository.png"
            The HACS "Custom repositories" dialog with the Roomer URL filled in and "Integration" selected as the type.

    5. Search HACS for **Roomer**, open it, and select **Download**.

        !!! example "Screenshot needed: hacs-roomer-download.png"
            The Roomer page in HACS with the Download button visible.

    6. Restart Home Assistant: **Settings → System**, the power icon in the top
       right, then **Restart Home Assistant**.

    HACS will tell you when a new version of Roomer is available.

=== "Manual"

    1. Download the latest release (or the repository as a ZIP) from
       [github.com/owens-ben/roomer](https://github.com/owens-ben/roomer).
    2. Find the `custom_components/roomer` folder inside it.
    3. Copy that folder into the `custom_components` folder of your Home
       Assistant configuration directory (the folder that holds
       `configuration.yaml`). Create `custom_components` if it doesn't exist.

        When you're done, this file should exist:

        ```text
        config/custom_components/roomer/manifest.json
        ```

        The Samba share, SSH, or the File editor add-on are easy ways to copy
        files onto Home Assistant OS.

    4. Restart Home Assistant: **Settings → System**, the power icon in the top
       right, then **Restart Home Assistant**.

    To update later, replace the folder with the new version and restart again.

## Check it worked

1. Go to **Settings → Devices & services**.
2. Select **Add integration** in the bottom right.
3. Search for **Roomer**.

If Roomer shows up in the list, it's installed. If it doesn't, refresh your
browser (Home Assistant caches the integration list), and check that the folder
is named exactly `roomer`.

!!! example "Screenshot needed: add-integration-search.png"
    The "Add integration" search dialog with "Roomer" typed in and the Roomer result shown.


Next: [First setup](first-setup.md).
